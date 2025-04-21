import argparse
import os
import sys
from functools import reduce
from collections import defaultdict

sys.path.append('..')

import numpy as np
from linearpost.dataset import Dataset
from linearpost import metrics

from linearpost.postprocess import LinearPostSimple


def main():
  args = parse_args()
  cache_dir = args.cache_dir
  cache_fnames = args.cache_fname
  criteria = args.criteria
  n_alphas = args.n_alphas
  iters_dcal = args.iters_dcal
  seed = args.seed
  results_dir = args.results_dir
  val_split = args.val_split
  eval_splits = args.eval_splits
  overwrite_results = args.overwrite_results
  bootstrap_n_resamples = args.bootstrap_n_resamples

  # Post-processing arguments that are not exposed
  solver = 'GUROBI'

  os.makedirs(results_dir, exist_ok=True)

  for cache_fname in cache_fnames:
    if not cache_fname.endswith('.pickle'):
      cache_fname += '.pickle'
    cache_path = os.path.join(cache_dir, cache_fname)
    if not os.path.exists(cache_path):
      raise FileNotFoundError(f"{cache_path} does not exist")
    cache_fname_base = os.path.splitext(cache_fname)[0]

    D_cache = Dataset.from_path(cache_path)
    n_classes = D_cache.features['labels'].n_categories
    n_groups = D_cache.features['groups'].n_categories

    print(f"Working on {cache_fname}" +
          (f" with {iters_dcal} decision calibration iteration(s)"
           if iters_dcal else ""))

    for criterion in criteria:

      ## Init (or load existing) metric loggers

      result_fname = f"{{split}}_{cache_fname_base}_{criterion}_linearpost{'_dcal'+str(iters_dcal) if iters_dcal else ''}.csv"
      result_path = os.path.join(results_dir, result_fname)
      loggers = {}
      for split in eval_splits:
        loggers[split] = metrics.MetricLogger.from_path(
            None if overwrite_results else result_path.format(split=split),
            n_classes=n_classes,
            n_groups=n_groups,
            return_std_err=True,
            n_resamples=bootstrap_n_resamples,
            random_state=seed,
        )

      ## Get alpha sweep from the fairness violation without post-proc to 0.001

      # Use val split to get the maximum fairness violation alpha_max
      if val_split is not None and val_split in D_cache.split:
        preds_val = D_cache.split[val_split]['p_y_x'].argmax(axis=1)
        metrics_baseline = metrics.evaluate(
            D_cache.split[val_split]['labels'],
            preds_val,
            D_cache.split[val_split]['groups'],
            n_classes=n_classes,
            n_groups=n_groups,
        )
        if criterion == 'tpr' and n_classes == 2:
          criterion_metric_name = 'tpr_binary_disparity'
        else:
          criterion_metric_name = f'{criterion}_disparity'
        alpha_max = metrics_baseline[criterion_metric_name]
      else:
        alpha_max = 1.0

      # Equally-spaced sweep, and include the unfair baseline float('inf')
      alphas = [float('inf')] + list(
          np.linspace(0.001, alpha_max, num=n_alphas).flatten())[:-1][::-1]

      # Remove alphas that have been previously processed and logged
      alphas_exist = defaultdict(list)
      alpha_isin = lambda a, a_exist: np.isclose(a, a_exist).any()
      if not overwrite_results:
        for split in eval_splits:
          if len(loggers[split]):
            alphas_exist[split] = loggers[split].df['alpha'].values.flatten()
        alphas_exist_all = reduce(
            np.intersect1d, [alphas_exist[split] for split in eval_splits])
        alphas = [a for a in alphas if not alpha_isin(a, alphas_exist_all)]

      if alphas:
        print(
            f"  - Post-processing for {criterion} with alpha: {', '.join([f'{alpha:.4f}' for alpha in alphas])} ",
            end='',
            flush=True)
      else:
        print(f"  - Skipping {criterion}, no new alpha to process")
        continue

      ## Post-process with LinearPost

      for alpha in alphas:
        postprocessor = LinearPostSimple(
            n_classes=n_classes,
            n_groups=n_groups,
            fairness_criterion=criterion,
            alpha=alpha,
            max_iters_dcal=iters_dcal,
            seed=seed,
        ).fit(
            p_a_x=D_cache.split['post']['p_a_x'],
            p_y_x=D_cache.split['post']['p_y_x'],
            p_ay_x=D_cache.split['post']['p_ay_x'],
            groups=D_cache.split['post']['groups'] if iters_dcal else None,
            labels_ay=(D_cache.split['post']['labels_ay']
                       if iters_dcal else None),
            solver=solver,
            solve_primal=True,
        )

        ## Evaluation and logging

        for split in eval_splits:
          if not alpha_isin(alpha, alphas_exist[split]):

            fair_preds = postprocessor.predict(
                p_a_x=D_cache.split[split]['p_a_x'],
                p_y_x=D_cache.split[split]['p_y_x'],
                p_ay_x=D_cache.split[split]['p_ay_x'])

            loggers[split].log_evaluate(
                D_cache.split[split]['labels'],
                fair_preds,
                D_cache.split[split]['groups'],
                alpha=alpha,
                seed=seed,
            )
            loggers[split].to_path(result_path.format(split=split))

        print('.', end='', flush=True)
      print(" Done")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache_dir", type=str, default="cache")
  parser.add_argument(
      "--cache_fname",
      type=str,
      nargs='+',
      required=True,
  )

  parser.add_argument(
      "--criteria",
      type=str,
      nargs='+',
      default=["sp", "tpr", "eo"],
      choices=["sp", "tpr", "fpr", "eo"],
  )
  parser.add_argument('--n_alphas', type=int, default=16)
  parser.add_argument('--iters_dcal', type=int, default=0)

  parser.add_argument("--seed", type=int, default=33)

  parser.add_argument("--results_dir", type=str, default="results")
  parser.add_argument("--val_split", type=str, default="val")
  parser.add_argument("--eval_splits",
                      type=str,
                      nargs='+',
                      default=['val', 'test'])

  parser.add_argument("--overwrite_results", action='store_true', default=False)
  parser.add_argument("--bootstrap_n_resamples", type=int, default=1000)

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main()
