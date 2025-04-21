import argparse
import os
import sys

sys.path.append('..')

import torch
import numpy as np
from tqdm import tqdm
import transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import step0_get_datasets


def fit(model,
        dataloader_train,
        label_column='labels',
        n_epochs=3,
        lr=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0):
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {
          "params": [
              p for n, p in model.named_parameters()
              if not any(nd in n for nd in no_decay)
          ],
          "weight_decay": weight_decay,
      },
      {
          "params": [
              p for n, p in model.named_parameters()
              if any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.0
      },
  ]
  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=(warmup_ratio * n_epochs * len(dataloader_train)),
      num_training_steps=n_epochs * len(dataloader_train))

  model_input_args = list(model.forward.__code__.co_varnames)
  loss_fn = torch.nn.CrossEntropyLoss()

  for epoch in range(n_epochs):
    model.train()
    for batch in tqdm(dataloader_train, desc=f"train epoch {epoch+1}"):
      batch = {k: v.to(model.device) for k, v in batch.items()}
      optimizer.zero_grad()
      outputs = model(**{
          k: v for k, v in batch.items() if k in model_input_args
      })
      loss = loss_fn(outputs.logits, batch[label_column])
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
      scheduler.step()


def predict_probas(model, dataloader):
  model_input_args = list(model.forward.__code__.co_varnames)
  model.eval()
  with torch.no_grad():
    probas = []
    for batch in tqdm(dataloader, desc="inference"):
      batch = {
          k: v.to(model.device)
          for k, v in batch.items()
          if k in model_input_args
      }
      outputs = model(**batch)
      probas.append(outputs.logits.softmax(dim=1).cpu().numpy())
    probas = np.concatenate(probas, axis=0)
    return probas


def main():
  args = parse_args()
  data_dir_base = args.data_dir_base
  dataset_getter_names = args.dataset_getter_name
  attr_aware = args.attr_aware
  models = args.model
  cache_pre = args.cache_pre
  cache_dir = args.cache_dir
  seed = args.seed
  device = args.device or 'cuda' if torch.cuda.is_available() else 'cpu'

  # Model training hyperparameters
  batch_size = args.batch_size
  n_epochs = args.n_epochs
  lr = args.lr
  warmup_ratio = args.warmup_ratio
  weight_decay = args.weight_decay
  max_grad_norm = args.max_grad_norm

  os.makedirs(data_dir_base, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  for dataset_getter_name in dataset_getter_names:
    for model_name in models:
      model_name_short = os.path.basename(model_name)
      cache_fname = f"{dataset_getter_name}_{model_name_short}.pickle"
      cache_path = os.path.join(cache_dir, cache_fname)

      print(
          f"Training {model_name_short} on {dataset_getter_name} (attribute-{'aware' if attr_aware else 'blind'}) and cache its predictions at:\n{cache_path}"
      )
      if os.path.exists(cache_path):
        print("  - Skipping, found existing cache")

      else:

        ## Prepare data

        # Load dataset
        D = step0_get_datasets.name_to_getter[dataset_getter_name](
            data_dir_base, seed=seed)
        n_classes = D.features['labels'].n_categories
        n_groups = D.features['groups'].n_categories

        # Get tokenizer and tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        X_tokenized = [
            tokenizer(text,
                      padding=False,
                      max_length=tokenizer.model_max_length,
                      truncation=True) for text in D['X']
        ]
        for k in X_tokenized[0].keys():
          D[k] = [x[k] for x in X_tokenized]

        ## Get and fine-tune model

        D['labels_ay'] = D['groups'] * n_classes + D['labels']
        if attr_aware:
          # Train Pr[ Y | X, A ] predictor (A is known and is part of X)
          label_column = 'labels'
          n_targets = n_classes
        else:
          # Train Pr[ A, Y | X ] predictor
          label_column = 'labels_ay'
          n_targets = n_classes * n_groups

        # Get model
        if seed is not None:
          transformers.set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_targets).to(device)

        # Fine-tune model
        if seed is not None:
          transformers.set_seed(seed)
        data_collator = transformers.DataCollatorWithPadding(tokenizer)
        dataloader_train = D.split['pre'].to_dataloader(
            column_names=([label_column] + list(X_tokenized[0].keys())),
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True)
        fit(model,
            dataloader_train,
            label_column=label_column,
            n_epochs=n_epochs,
            lr=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm)

        ## Get predictions

        if not cache_pre:
          # Get a subset of D that excludes the pre-train split
          D_cache = D.split[['post', 'val', 'test']]
        else:
          D_cache = D

        dataloader_cache = D_cache.to_dataloader(
            column_names=list(X_tokenized[0].keys()),
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        if attr_aware:
          # Get predicted Pr[ Y | X, A ] on D_cache
          D_cache['p_y_x'] = predict_probas(model, dataloader_cache)

          # Pr[ A | X, A ] is one-hot vector
          D_cache['p_a_x'] = np.eye(n_groups)[D_cache['groups']]
          # Pr[ A, Y | X, A ] = Pr[ A | X, A ] * Pr[ Y | X, A ]
          D_cache['p_ay_x'] = (D_cache['p_a_x'][:, :, None] *
                               D_cache['p_y_x'][:, None, :]).reshape(
                                   -1, n_groups * n_classes)
        else:
          # Get predicted Pr[ A, Y | X ] on D_cache
          D_cache['p_ay_x'] = predict_probas(model, dataloader_cache)

          # Marginalize to get Pr[ A | X ] and Pr[ Y | X ]
          D_cache['p_a_x'] = D_cache['p_ay_x'].reshape(-1, n_groups,
                                                       n_classes).sum(axis=2)
          D_cache['p_y_x'] = D_cache['p_ay_x'].reshape(-1, n_groups,
                                                       n_classes).sum(axis=1)

        D_cache.to_path(cache_path)
        print(f"  - Cached dataset to {cache_path}")


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--data_dir_base", type=str, default="../data")
  parser.add_argument(
      "--dataset_getter_name",
      type=str,
      nargs='+',
      required=True,
      choices=list(step0_get_datasets.name_to_getter.keys()),
  )
  parser.add_argument('--attr_aware', action='store_true', default=False)

  parser.add_argument(
      "--model",
      type=str,
      nargs='+',
      default=["google-bert/bert-base-uncased"],
  )

  parser.add_argument('--cache_pre', action='store_true', default=False)
  parser.add_argument("--cache_dir", type=str, default="cache")

  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--device", type=str, default=None)

  # Model fine-tuning hyperparameters
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--n_epochs", type=int, default=3)
  parser.add_argument("--lr", type=float, default=2e-5)
  parser.add_argument("--warmup_ratio", type=float, default=0.1)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main()
