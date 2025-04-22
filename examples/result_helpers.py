import os
import sys
from functools import partial
from collections import defaultdict

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

import linearpost.metrics as metrics
from linearpost.metrics import MetricLogger


def plot(ax,
         result_path,
         fairness_criterion,
         weighted_fairness_criterion=False,
         performance_metric='accuracy',
         set_xlabel=False,
         set_ylabel=False,
         **kwargs):
  df_val = MetricLogger.from_csv(result_path.format(split='val')).df
  df_test = MetricLogger.from_csv(result_path.format(split='test')).df

  fairness_criterion = f'{fairness_criterion}_disparity'
  if fairness_criterion == 'tpr_disparity' and fairness_criterion not in df_val:
    fairness_criterion = 'tpr_binary_disparity'
  if weighted_fairness_criterion:
    fairness_criterion += '_weighted'

  i_unique = metrics.get_unique_idx(df_val,
                                    fairness_criterion,
                                    performance_metric=performance_metric,
                                    subset=np.arange(len(df_test)))
  i_pareto = metrics.get_pareto_idx(df_val,
                                    fairness_criterion,
                                    performance_metric=performance_metric,
                                    tolerance=0.1,
                                    subset=i_unique)
  if 0 not in i_pareto:
    i_pareto = np.insert(i_pareto, 0, 0)

  df_val = df_val.iloc[i_pareto]
  df_test = df_test.iloc[i_pareto]

  y = df_test[performance_metric]['mean'].values
  x = df_test[fairness_criterion]['mean'].values
  yerr = (None if 'std' not in df_test[performance_metric] else
          df_test[performance_metric]['std'].values)
  xerr = (None if 'std' not in df_test[fairness_criterion] else
          df_test[fairness_criterion]['std'].values)

  # ax.errorbar(x, y, xerr=xerr, yerr=yerr)
  ret = ax.plot(x, y, **kwargs)

  if set_ylabel:
    ax.set_ylabel(performance_metric)
  if set_xlabel:
    ax.set_xlabel(fairness_criterion)
  return ret


# Xian and Zhao, 2024

xz24_dataset_names = ['adult', 'compas', 'acsincome2', 'acsincome5', 'biasbios']
xz24_blind_only = ['biasbios']
xz24_models = ['logreg', 'lgbm', 'mlp', 'bert-base-uncased']
xz24_methods = ['linearpost', 'linearpost_dcal10']
xz24_criteria = ['sp', 'tpr', 'eo']
xz24_n_alphas = 16


def get_xz24_result_paths(dataset_names=None):

  def get_result_path(dataset_name,
                      model,
                      method,
                      criterion,
                      aware=False,
                      results_dir='results'):
    if dataset_name == 'biasbios':
      result_fname = f"{{split}}_xz24_{dataset_name}_{model}_{criterion}_{method}.csv"
    else:
      result_fname = f"{{split}}_xz24_{dataset_name}_{'aware' if aware else 'blind'}_{model}_{criterion}_{method}.csv"
    result_path = os.path.join(results_dir, result_fname)
    return result_path

  if dataset_names is None:
    dataset_names = xz24_dataset_names

  # result_paths[dataset_name][model][aware][method][criterion]
  result_paths = defaultdict(
      lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

  for dataset_name in dataset_names:
    for aware in [False, True]:
      if dataset_name in xz24_blind_only:
        if aware:
          continue
      for model in xz24_models:
        for method in xz24_methods:
          for criterion in xz24_criteria:
            result_path = get_result_path(dataset_name,
                                          model,
                                          method,
                                          criterion,
                                          aware=aware)
            if os.path.exists(result_path.format(split='test')):
              result_paths[dataset_name][model][aware][method][
                  criterion] = result_path
  return result_paths


def plot_xz24_dataset(dataset_name,
                      performance_metric='accuracy',
                      weighted_fairness_criterion=False):
  result_paths = get_xz24_result_paths([dataset_name])

  styles = {
      ('linearpost', 'blind'): {
          'color': '#866A9A',
          'lw': 2.5,
          'marker': 'o',
          'markersize': 3.5,
          'zorder': 23,
      },
      ('linearpost_dcal10', 'blind'): {
          'color': '#C9B2D8',
          'lw': 1.5,
          'marker': 'd',
          'markersize': 5.2,
          'zorder': 21,
      },
      ('linearpost', 'aware'): {
          'color': '#C79800',
          'lw': 2.5,
          'marker': 's',
          'markersize': 3.2,
          'zorder': 22,
      },
      ('linearpost_dcal10', 'aware'): {
          'color': '#F5D87A',
          'lw': 1.5,
          'marker': '^',
          'markersize': 5.6,
          'zorder': 20,
      },
  }
  labels = {
      ('linearpost', 'blind'): 'blind, linearpost',
      ('linearpost_dcal10', 'blind'): 'blind, linearpost + dcal',
      ('linearpost', 'aware'): 'aware, linearpost',
      ('linearpost_dcal10', 'aware'): 'aware, linearpost + dcal',
  }
  handles = {}

  # Get the number of rows for the plot
  models = [m for m in xz24_models if m in result_paths[dataset_name]]

  fig, axs = plt.subplots(
      len(models),
      3,
      figsize=(8, 0.5 + 2.3 * len(models)),
      # sharex='col',
      # sharey='row',
      dpi=300)

  for i, model in enumerate(models):
    for j, criterion in enumerate(xz24_criteria):
      axs.flat[i * 3 + j].set_title(f'{model} / {criterion}', fontsize=10)

      for method in xz24_methods:
        for aware in [False, True]:
          try:
            result_path = result_paths[dataset_name][model][aware][method][
                criterion]
          except KeyError:
            continue
          style = styles[(method, 'aware' if aware else 'blind')]
          plot_fn = partial(
              plot,
              axs.flat[i * 3 + j],
              result_path,
              criterion,
              performance_metric=performance_metric,
              weighted_fairness_criterion=weighted_fairness_criterion,
              set_xlabel=i == len(models) - 1,
              color=style['color'])
          line, = plot_fn(lw=1, alpha=0.6, zorder=style['zorder'])
          markers, = plot_fn(linestyle='None',
                             marker=style['marker'],
                             markersize=style['markersize'] * 1.3,
                             markeredgewidth=0.6,
                             markeredgecolor='white',
                             zorder=style['zorder'] + 50)
          handles[(method, 'aware' if aware else 'blind')] = (line, markers)

  fig.tight_layout()
  fig.suptitle(dataset_name, fontsize=10, y=1.03)
  fig.supylabel(performance_metric, fontsize=10, x=-0.03)
  fig.legend([handles[k] for k in labels if k in handles],
             [labels[k] for k in labels if k in handles],
             loc='upper center',
             ncol=2,
             fontsize=10,
             frameon=False,
             bbox_to_anchor=(0.5, -0.0))

  for ax in axs.flat:
    ax.spines['top'].set_zorder(100)
    ax.spines['bottom'].set_zorder(100)
    ax.spines['left'].set_zorder(100)
    ax.spines['right'].set_zorder(100)

  return fig, axs
