import argparse
import os
import sys

sys.path.append('..')

import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from linearpost.models import MLPClassifier

import step0_get_datasets


def get_model(name, device=None, seed=None):
  if name == 'logreg':
    return LogisticRegression(max_iter=10000, random_state=seed)
  elif name == 'lgbm':
    return LGBMClassifier(random_state=seed, verbosity=0)
  elif name == 'mlp':
    return MLPClassifier(
        hidden_layer_sizes=(500, 200, 100),
        n_epochs=20,
        batch_size=128,
        lr=1e-3,
        gamma=0.8,
        random_state=seed,
        device=device,
    )
  else:
    raise NotImplementedError


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

  os.makedirs(data_dir_base, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  for dataset_getter_name in dataset_getter_names:
    for model in models:
      cache_fname = f"{dataset_getter_name}_{model}.pickle"
      cache_path = os.path.join(cache_dir, cache_fname)

      print(
          f"Training {model} on {dataset_getter_name} (attribute-{'aware' if attr_aware else 'blind'}) and cache its predictions at:\n{cache_path}"
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

        ## Get and train model

        D['labels_ay'] = D['groups'] * n_classes + D['labels']
        if attr_aware:
          # Train Pr[ Y | X, A ] predictor (A is known and is part of X)
          label_column = 'labels'
          n_targets = n_classes
        else:
          # Train Pr[ A, Y | X ] predictor
          label_column = 'labels_ay'
          n_targets = n_classes * n_groups

        predictor = get_model(model, device=device, seed=seed)
        predictor.fit(D.split['pre']['X'], D.split['pre'][label_column])

        ## Get predictions

        if not cache_pre:
          # Get a subset of D that excludes the pre-train split
          D_cache = D.split[['post', 'val', 'test']]
        else:
          D_cache = D

        if attr_aware:
          # Get predicted Pr[ Y | X, A ] on D_cache
          D_cache['p_y_x'] = predictor.predict_proba(D_cache['X'])

          # Pr[ A | X, A ] is one-hot vector
          D_cache['p_a_x'] = np.eye(n_groups)[D_cache['groups']]
          # Pr[ A, Y | X, A ] = Pr[ A | X, A ] * Pr[ Y | X, A ]
          D_cache['p_ay_x'] = (D_cache['p_a_x'][:, :, None] *
                               D_cache['p_y_x'][:, None, :]).reshape(
                                   -1, n_groups * n_classes)
        else:
          # Get predicted Pr[ A, Y | X ] on D_cache
          D_cache['p_ay_x'] = predictor.predict_proba(D_cache['X'])

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
      required=True,
      choices=["logreg", "lgbm", "mlp"],
  )

  parser.add_argument('--cache_pre', action='store_true', default=False)
  parser.add_argument("--cache_dir", type=str, default="cache")

  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--device", type=str, default=None)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main()
