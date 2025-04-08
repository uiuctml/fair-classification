import sys

sys.path.append('..')

import argparse
import os
import pickle

import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from linearpost.models import MLPClassifier

from linearpost.dataset import Dataset
from linearpost import loader
from linearpost import postprocess
from linearpost import metrics


def get_dataset(name, data_dir_base, remove_sensitive_attr=False, seed=None):
  data_dir = os.path.join(data_dir_base, name)

  if name == 'adult':
    sensitive_attr = 'sex'
    loader_outputs = loader.adult(
        data_dir,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.3, 0.2, 0.2, 0.3]

  elif name == 'compas':
    sensitive_attr = 'race'
    loader_outputs = loader.compas(
        data_dir,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.3, 0.2, 0.2, 0.3]

  elif name == 'acsincome2':
    sensitive_attr = 'SEX'
    data_dir = os.path.join(data_dir_base, 'acsincome')
    loader_outputs = loader.acsincome(
        data_dir,
        n_classes=2,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.5, 0.1, 0.1, 0.3]

  elif name == 'acsincome5':
    sensitive_attr = 'RAC1P'
    data_dir = os.path.join(data_dir_base, 'acsincome')
    loader_outputs = loader.acsincome(
        data_dir,
        n_classes=5,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.5, 0.1, 0.1, 0.3]

    # Combine RAC1P categories 3, 4, 5, and 6, 7, and 8, 9 into new categories
    # 10, 11, and 12 respectively, due to small sample size in some groups.
    # This is also consistent with the UCI Adult dataset.
    category_names = loader_outputs['category_names']
    category_names['RAC1P']['9997'] = "American Indian or Alaska Native alone"
    category_names['RAC1P'][
        '9998'] = "Asian, Native Hawaiian or Other Pacific Islander alone"
    category_names['RAC1P']['9999'] = "Other"

    df = loader_outputs['data']
    df['RAC1P'] = df['RAC1P'].astype(float)
    df['RAC1P'] = df['RAC1P'].replace([3.0, 4.0, 5.0], 9997.0)
    df['RAC1P'] = df['RAC1P'].replace([6.0, 7.0], 9998.0)
    df['RAC1P'] = df['RAC1P'].replace([8.0, 9.0], 9999.0)
    df['RAC1P'] = df['RAC1P'].astype('category')

    # Get group labels
    groups = df['RAC1P'].values
    group_names, groups = np.unique(groups, return_inverse=True)
    loader_outputs['groups'] = groups
    loader_outputs['group_names'] = [
        category_names['RAC1P'][str(int(n))] for n in group_names
    ]

  D = Dataset.from_loader_outputs(loader_outputs)
  D.create_splits(
      split_sizes,
      ['pre', 'post', 'val', 'test'],
      shuffle=True,
      seed=seed,
  )
  if remove_sensitive_attr:
    D['X'].drop([sensitive_attr], inplace=True, axis=1)
  print('Dataset columns:', D['X'].columns.tolist())
  D.preprocess_tabular('X', train_split_name='pre')
  return D


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


def main():
  args = parse_args()
  dataset_name = args.dataset_name
  data_dir_base = args.data_dir_base
  criteria = args.criteria
  models = args.models
  results_dir = args.results_dir
  cache_dir = args.cache_dir
  seed = args.seed
  device = args.device or 'cuda' if torch.cuda.is_available() else 'cpu'
  attribute_awareness = [
      x for x in [args.attr_aware, args.attr_blind] if x is not None
  ]
  if not attribute_awareness:
    attribute_awareness = [True, False]

  # Postprocessor arguments that are not exposed
  solver = 'GUROBI'

  os.makedirs(data_dir_base, exist_ok=True)
  os.makedirs(results_dir, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  for aware in attribute_awareness:
    for model in models:
      for criterion in criteria:
        print(
            f"Working on {dataset_name} with {model} for {criterion} ({'aware' if aware else 'blind'})"
        )

        cache_fname = f"{dataset_name}_{'aware' if aware else 'blind'}_{model}.pickle"
        cache_path = os.path.join(cache_dir, cache_fname)
        if os.path.exists(cache_path):
          D_post = pickle.load(open(cache_path, 'rb'))
          n_classes = D_post.features['labels'].n_categories
          n_groups = D_post.features['groups'].n_categories
          print(f"Loaded cached dataset from {cache_path}")

        else:
          D = get_dataset(dataset_name,
                          data_dir_base,
                          remove_sensitive_attr=not aware,
                          seed=seed)
          predictor = get_model(model, device=device, seed=seed)

          n_classes = D.features['labels'].n_categories
          n_groups = D.features['groups'].n_categories
          D['labels_ay'] = D['groups'] * n_classes + D['labels']

          # Drop pretrain split from dataset
          D_post = Dataset.from_splits(D.get_splits(['post', 'val', 'test']))

          if aware:
            # Train Pr[ Y | X ] predictor
            predictor.fit(D.get_split('pre')['X'], D.get_split('pre')['labels'])
            D_post['p_y_x'] = predictor.predict_proba(D_post['X'])
            D_post['p_a_x'] = np.eye(n_groups)[D_post['groups']]
            D_post['p_ay_x'] = (D_post['p_a_x'][:, :, None] *
                                D_post['p_y_x'][:, None, :]).reshape(
                                    -1, n_groups * n_classes)
          else:
            # Train Pr[ A, Y | X ] predictor
            predictor.fit(
                D.get_split('pre')['X'],
                D.get_split('pre')['labels_ay'])
            D_post['p_ay_x'] = predictor.predict_proba(D_post['X'])
            D_post['p_a_x'] = D_post['p_ay_x'].reshape(-1, n_groups,
                                                       n_classes).sum(axis=2)
            D_post['p_y_x'] = D_post['p_ay_x'].reshape(-1, n_groups,
                                                       n_classes).sum(axis=1)

          with open(cache_path, 'wb') as f:
            pickle.dump(D_post, f)
            print(f"Cached dataset to {cache_path}")

        result_fname = f"{{split}}_linearpost_{dataset_name}_{'aware' if aware else 'blind'}_{model}_{criterion}.csv"
        result_path = os.path.join(results_dir, result_fname)
        loggers = {}
        for split in ['val', 'test']:
          loggers[split] = metrics.MetricLogger(
              n_classes=n_classes,
              n_groups=n_groups,
              return_std_err=True,
              random_state=seed,
          )

        preds_val = D_post.get_split('val')['p_y_x'].argmax(axis=1)
        metrics_baseline = metrics.evaluate(
            D_post.get_split('val')['labels'],
            preds_val,
            D_post.get_split('val')['groups'],
            n_classes=n_classes,
            n_groups=n_groups,
        )
        alpha_max = metrics_baseline[f'{criterion}_disparity']
        alphas = [float('inf')] + list(
            np.linspace(0.001, alpha_max, num=16).flatten())[:-1][::-1]

        for alpha in alphas:
          postprocessor = postprocess.LinearPostSimple(
              n_classes=n_classes,
              n_groups=n_groups,
              fairness_criterion=criterion,
              alpha=alpha,
              seed=seed,
          ).fit(p_a_x=D_post.get_split('post')['p_a_x'],
                p_y_x=D_post.get_split('post')['p_y_x'],
                p_ay_x=D_post.get_split('post')['p_ay_x'],
                solver=solver,
                solve_primal=True)

          fair_preds_val = postprocessor.predict(
              p_a_x=D_post.get_split('val')['p_a_x'],
              p_y_x=D_post.get_split('val')['p_y_x'],
              p_ay_x=D_post.get_split('val')['p_ay_x'])
          fair_preds_test = postprocessor.predict(
              p_a_x=D_post.get_split('test')['p_a_x'],
              p_y_x=D_post.get_split('test')['p_y_x'],
              p_ay_x=D_post.get_split('test')['p_ay_x'])

          for split, preds in zip(['val', 'test'],
                                  [fair_preds_val, fair_preds_test]):
            loggers[split].log_evaluate(
                D_post.get_split(split)['labels'],
                preds,
                D_post.get_split(split)['groups'],
                alpha=alpha,
                seed=seed,
            )
            loggers[split].to_csv(result_path.format(split=split))


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dataset_name",
      type=str,
      required=False,
      choices=["adult", "acsincome2", "acsincome5", "compas"],
  )
  parser.add_argument("--data_dir_base", type=str, required=False)
  parser.add_argument(
      "--criteria",
      type=str,
      nargs='+',
      default=["sp", "tpr", "eo"],
      choices=["sp", "tpr", "fpr", "eo"],
  )
  parser.add_argument(
      "--models",
      type=str,
      nargs='+',
      default=["logreg", "lgbm", "mlp"],
      choices=["logreg", "lgbm", "mlp"],
  )
  parser.add_argument('--attr_aware', action='store_true', default=None)
  parser.add_argument('--attr_blind', action='store_false', default=None)
  parser.add_argument("--seed", type=int, default=33)
  parser.add_argument("--device", type=str, default=None)

  parser.add_argument("--results_dir", type=str, default="results")
  parser.add_argument("--cache_dir", type=str, default="cache")

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main()
