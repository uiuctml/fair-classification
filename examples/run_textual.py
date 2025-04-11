import sys

sys.path.append('..')

import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from linearpost.dataset import Dataset
from linearpost import loader
from linearpost import postprocess
from linearpost import metrics


def get_dataset(name, data_dir_base, seed=None):
  data_dir = os.path.join(data_dir_base, name)

  if name == 'biasbios':
    loader_outputs = loader.biasbios(data_dir)
    split_sizes = [0.5, 0.1, 0.1, 0.3]

  D = loader.dataset_from_loader_outputs(loader_outputs)
  D.create_splits(
      split_sizes,
      ['pre', 'post', 'val', 'test'],
      shuffle=True,
      seed=seed,
  )
  return D


def get_model(model_name, n_classes, device=None, seed=None):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if seed is not None:
    transformers.set_seed(seed)
  model = AutoModelForSequenceClassification.from_pretrained(
      model_name, num_labels=n_classes).to(device)
  return model, tokenizer


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
  dataset_name = args.dataset_name
  data_dir_base = args.data_dir_base
  model_names = args.model_names
  criteria = args.criteria
  n_alphas = args.n_alphas
  cache_dir = args.cache_dir
  results_dir = args.results_dir
  overwrite_results = args.overwrite_results
  seed = args.seed
  device = args.device or 'cuda' if torch.cuda.is_available() else 'cpu'

  # Model training hyperparameters that are not exposed
  batch_size = 32
  n_epochs = 3
  lr = 2e-5
  warmup_ratio = 0.1
  weight_decay = 0.01
  max_grad_norm = 1.0

  # Postprocessor arguments that are not exposed
  solver = 'GUROBI'

  os.makedirs(data_dir_base, exist_ok=True)
  os.makedirs(results_dir, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  for model_name in model_names:
    model_name_short = os.path.basename(model_name)
    for criterion in criteria:
      print(
          f"Working on {dataset_name} for {criterion} with {model_name_short}")

      cache_fname = f"{dataset_name}_{model_name_short}.pickle"
      cache_path = os.path.join(cache_dir, cache_fname)
      if os.path.exists(cache_path):
        D_post = Dataset.from_path(cache_path)
        n_classes = D_post.features['labels'].n_categories
        n_groups = D_post.features['groups'].n_categories
        print(f"  - Loaded cached dataset from {cache_path}")

      else:
        D = get_dataset(dataset_name, data_dir_base, seed=seed)

        n_classes = D.features['labels'].n_categories
        n_groups = D.features['groups'].n_categories
        D['labels_ay'] = D['groups'] * n_classes + D['labels']

        # Load model for Pr[ A, Y | X ] predictions
        model, tokenizer = get_model(model_name,
                                     n_classes=n_classes * n_groups,
                                     device=device,
                                     seed=seed)
        data_collator = transformers.DataCollatorWithPadding(tokenizer)

        # Tokenize dataset
        X_tokenized = [
            tokenizer(text,
                      padding=False,
                      max_length=tokenizer.model_max_length,
                      truncation=True) for text in D['X']
        ]
        for k in X_tokenized[0].keys():
          D[k] = [x[k] for x in X_tokenized]

        # Train Pr[ A, Y | X ] predictor
        dataloader_train = D.split['pre'].to_dataloader(
            column_names=(['labels_ay'] + list(X_tokenized[0].keys())),
            batch_size=batch_size,
            collate_fn=data_collator)
        fit(model,
            dataloader_train,
            label_column='labels_ay',
            n_epochs=n_epochs,
            lr=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm)

        # Drop pretrain split from dataset
        D_post = D.split[['post', 'val', 'test']]

        dataloader_cache = D_post.to_dataloader(
            column_names=list(X_tokenized[0].keys()),
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        D_post['p_ay_x'] = predict_probas(model, dataloader_cache)
        D_post['p_a_x'] = D_post['p_ay_x'].reshape(-1, n_groups,
                                                   n_classes).sum(axis=2)
        D_post['p_y_x'] = D_post['p_ay_x'].reshape(-1, n_groups,
                                                   n_classes).sum(axis=1)

        D_post.to_path(cache_path)
        print(f"  - Cached dataset to {cache_path}")

      result_fname = f"{{split}}_linearpost_{dataset_name}_{model_name_short}_{criterion}.csv"
      result_path = os.path.join(results_dir, result_fname)
      loggers = {}
      for split in ['val', 'test']:
        if overwrite_results:
          loggers[split] = metrics.MetricLogger(
              n_classes=n_classes,
              n_groups=n_groups,
              return_std_err=True,
              random_state=seed,
          )
        else:
          loggers[split] = metrics.MetricLogger.from_path(
              result_path.format(split=split),
              n_classes=n_classes,
              n_groups=n_groups,
              return_std_err=True,
              random_state=seed,
          )

      preds_val = D_post.split['val']['p_y_x'].argmax(axis=1)
      metrics_baseline = metrics.evaluate(
          D_post.split['val']['labels'],
          preds_val,
          D_post.split['val']['groups'],
          n_classes=n_classes,
          n_groups=n_groups,
      )
      if criterion == 'tpr' and n_classes == 2:
        criterion_metric_name = 'tpr_binary_disparity'
      else:
        criterion_metric_name = f'{criterion}_disparity'
      alpha_max = metrics_baseline[criterion_metric_name]
      alphas = [float('inf')] + list(
          np.linspace(0.001, alpha_max, num=n_alphas).flatten())[:-1][::-1]

      alphas_val_exist = np.array([])
      alphas_test_exist = np.array([])
      alpha_isin = lambda alpha, exist: np.isclose(alpha, exist).any()
      if not overwrite_results and len(loggers['val']) and len(loggers['test']):
        alphas_val_exist = loggers['val'].df['alpha'].values.flatten()
        alphas_test_exist = loggers['test'].df['alpha'].values.flatten()
        alphas_exist = np.array(
            list(set(alphas_val_exist) & set(alphas_test_exist)))
        alphas = [a for a in alphas if not alpha_isin(a, alphas_exist)]

      if alphas:
        print(
            f"  - Post-processing with alpha: {', '.join([f'{alpha:.4f}' for alpha in alphas])} ",
            end='',
            flush=True)
      else:
        print("  - Skipping, no new alpha to process")

      for alpha in alphas:
        postprocessor = postprocess.LinearPostSimple(
            n_classes=n_classes,
            n_groups=n_groups,
            fairness_criterion=criterion,
            alpha=alpha,
            seed=seed,
        ).fit(p_a_x=D_post.split['post']['p_a_x'],
              p_y_x=D_post.split['post']['p_y_x'],
              p_ay_x=D_post.split['post']['p_ay_x'],
              solver=solver,
              solve_primal=True)

        fair_preds_val = postprocessor.predict(
            p_a_x=D_post.split['val']['p_a_x'],
            p_y_x=D_post.split['val']['p_y_x'],
            p_ay_x=D_post.split['val']['p_ay_x'])
        fair_preds_test = postprocessor.predict(
            p_a_x=D_post.split['test']['p_a_x'],
            p_y_x=D_post.split['test']['p_y_x'],
            p_ay_x=D_post.split['test']['p_ay_x'])

        for split, preds in zip(['val', 'test'],
                                [fair_preds_val, fair_preds_test]):
          if not alpha_isin(
              alpha, alphas_val_exist if split == 'val' else alphas_test_exist):
            loggers[split].log_evaluate(
                D_post.split[split]['labels'],
                preds,
                D_post.split[split]['groups'],
                alpha=alpha,
                seed=seed,
            )
            loggers[split].to_path(result_path.format(split=split))

        print('.', end='', flush=True)

      if alphas:
        print(" done")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dataset_name",
      type=str,
      required=False,
      choices=["biasbios"],
  )
  parser.add_argument("--data_dir_base", type=str, required=False)
  parser.add_argument(
      "--model_names",
      type=str,
      nargs='+',
      default=["google-bert/bert-base-uncased"],
  )
  parser.add_argument(
      "--criteria",
      type=str,
      nargs='+',
      default=["sp", "tpr", "eo"],
      choices=["sp", "tpr", "fpr", "eo"],
  )
  parser.add_argument('--n_alphas', type=int, default=16)

  parser.add_argument("--seed", type=int, default=33)
  parser.add_argument("--device", type=str, default=None)

  parser.add_argument("--cache_dir", type=str, default="cache")
  parser.add_argument("--results_dir", type=str, default="results")
  parser.add_argument("--overwrite_results", action='store_true', default=False)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main()
