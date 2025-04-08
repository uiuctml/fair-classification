from functools import partial
from typing import Callable, Optional, Tuple
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics


def bootstrap_std_error(
    metric_fn: Callable[..., float]
) -> Callable[..., Tuple[float] | Tuple[float, float]]:
  """Wraps a metric function to compute bootstrap standard error if requested."""

  def wrapped(*args,
              return_std_err: bool = False,
              n_resamples: int = 1000,
              random_state: Optional[int] = None,
              **kwargs):
    res = (metric_fn(*args, **kwargs),)
    if return_std_err:
      bootstrap = scipy.stats.bootstrap(args,
                                        partial(metric_fn, **kwargs),
                                        n_resamples=n_resamples,
                                        paired=True,
                                        random_state=random_state,
                                        method='basic')
      res += (bootstrap.standard_error,)
    return res

  return wrapped


def accuracy(y_true, y_preds, return_std_err=False):
  """Compute accuracy."""
  correct = y_true == y_preds
  accuracy = np.mean(correct)
  res = (accuracy,)
  if return_std_err:
    std_err = scipy.stats.sem(correct)
    res += (std_err,)
  return res


def output_dists(y_preds, groups, n_classes, n_groups):
  group_counts = np.bincount(groups, minlength=n_groups)  # shape = (n_groups,)
  g_p, counts = np.unique(np.stack((groups, y_preds), axis=1),
                          axis=0,
                          return_counts=True)
  dists = np.zeros((n_groups, n_classes), dtype=float)
  g, p = g_p.T
  dists[g, p] = counts / group_counts[g]  # normalize
  return (dists, group_counts / group_counts.sum())


def confusion_matrix(y_true, y_preds, groups, n_classes, n_groups):
  g_y_p, counts = np.unique(np.stack((groups, y_true, y_preds), axis=1),
                            axis=0,
                            return_counts=True)
  cm = np.zeros((n_groups, n_classes, n_classes), dtype=float)
  g, y, p = g_y_p.T
  cm[g, y, p] = counts
  g_y_counts = cm.sum(axis=2)  # shape = (n_groups, n_classes)
  cm = np.divide(cm,
                 g_y_counts[..., None],
                 out=np.zeros_like(cm),
                 where=g_y_counts[..., None] != 0)  # avoid division by zero
  return (cm, g_y_counts / g_y_counts.sum())


def disparity(dists, weights, group_weights, weighted_sum=False, ord=np.inf):
  n_groups = dists.shape[0]
  if weighted_sum:
    dist_all = (np.moveaxis(dists, 0, -1) * group_weights).sum(axis=-1)
    all_diffs = dists - dist_all
    diffs = np.nan_to_num(np.linalg.norm(all_diffs, ord=ord, axis=-1), nan=0.0)
    return (weights * diffs).sum()
  else:
    # Pairwise differences
    all_diffs = dists[:, None, ...] - dists[None, :, ...]
    all_diffs = all_diffs.reshape(n_groups, n_groups, -1)
    diffs = np.nan_to_num(np.linalg.norm(all_diffs, ord=ord, axis=-1), nan=0.0)
    return diffs.max()


@bootstrap_std_error
def sp_disparity(y_preds,
                 groups,
                 n_classes,
                 n_groups,
                 weighted_sum=False,
                 ord=np.inf):
  return disparity(
      *output_dists(y_preds, groups, n_classes, n_groups),
      group_weights=np.bincount(groups, minlength=n_groups) / len(groups),
      weighted_sum=weighted_sum,
      ord=ord,
  )


@bootstrap_std_error
def eo_disparity(y_true,
                 y_preds,
                 groups,
                 n_classes,
                 n_groups,
                 weighted_sum=False,
                 ord=np.inf):
  return disparity(
      *confusion_matrix(y_true, y_preds, groups, n_classes, n_groups),
      group_weights=np.bincount(groups, minlength=n_groups) / len(groups),
      weighted_sum=weighted_sum,
      ord=ord,
  )


def tpr_disparity_dists(y_true, y_preds, groups, n_classes, n_groups):
  cm, g_y_dists = confusion_matrix(y_true, y_preds, groups, n_classes, n_groups)
  tpr = np.array([np.diag(c) for c in cm])[..., None]
  # tpr.shape = (n_groups, n_classes, 1)
  if n_classes == 2:
    tpr = tpr[:, 1, :]  # take the positive class, shape = (n_groups, 1)
    g_y_dists = g_y_dists[:, 1]  # shape = (n_groups,)
  return tpr, g_y_dists


@bootstrap_std_error
def tpr_disparity(y_true,
                  y_preds,
                  groups,
                  n_classes,
                  n_groups,
                  weighted_sum=False,
                  ord=np.inf):
  return disparity(
      *tpr_disparity_dists(y_true, y_preds, groups, n_classes, n_groups),
      group_weights=np.bincount(groups, minlength=n_groups) / len(groups),
      weighted_sum=weighted_sum,
      ord=ord,
  )


def fpr_disparity_dists(y_true, y_preds, groups, n_groups):
  cm, g_y_dists = confusion_matrix(y_true, y_preds, groups, 2, n_groups)
  return cm[:, 0, 1][..., None], g_y_dists[:, 0]


@bootstrap_std_error
def fpr_disparity(y_true,
                  y_preds,
                  groups,
                  n_classes,
                  n_groups,
                  weighted_sum=False,
                  ord=np.inf):
  assert n_classes == 2
  return disparity(
      *fpr_disparity_dists(y_true, y_preds, groups, n_groups),
      group_weights=np.bincount(groups, minlength=n_groups) / len(groups),
      weighted_sum=weighted_sum,
      ord=ord,
  )


def edgewise_simplex_subdivision(x, k=1):
  n = len(x)
  d = x.shape[1]

  cumsum = np.cumsum(x * k, axis=1)

  # get the boundaries and keep their associated colors
  boundaries = np.where(cumsum[:, :-1] != k, cumsum[:, :-1] % 1, 1)
  # last color should have cumsum = 1
  boundaries = np.concatenate([boundaries, np.ones((n, 1))], axis=1)
  boundaries_c = np.tile(np.arange(d), (n, 1))

  # sort the boundaries
  I = np.argsort(boundaries, axis=1, stable=True)
  boundaries = np.take_along_axis(boundaries, I, axis=1)
  boundaries_c = np.take_along_axis(boundaries_c, I, axis=1)

  # cutoffs at which a color is yield
  cutoffs = boundaries[:, None, :] + np.arange(k)[None, :, None]
  cutoffs = cutoffs.reshape(n, -1)
  # in case some rows do not sum to 1
  cutoffs = np.concatenate([cutoffs[:, :-1], cumsum[:, -1][:, None]], axis=1)

  # count the number of occurrences of each color
  counts = [np.zeros(n, dtype=int)]

  for i in range(d):

    # cumulative count for colors 1 to i
    c = (cutoffs < cumsum[:, i][:, None]).sum(axis=1)

    # handle ties
    ties = cutoffs == cumsum[:, i][:, None]
    ind = np.where(boundaries_c == i)[1]
    mask = np.arange(d)[None, :] <= ind[:, None]
    c += (ties & np.tile(mask, (1, k))).sum(axis=1)

    counts.append(c - counts.pop(-1))
    counts.append(c)  # also remember the cumulative count

  counts = counts[:-1]

  # C[j, i] is the number of times color i appears in M[j], the color scheme
  # of the j-th example
  C = np.stack(counts, axis=1)
  return C


def binned_calibration_error(probas, y_true, k=2):
  n = len(probas)
  n_classes = probas.shape[1]
  z = edgewise_simplex_subdivision(probas, k)
  _, b = np.unique(z, axis=0, return_inverse=True)
  ece = 0
  for i in range(max(b)):
    f = probas[b == i].sum(axis=0)
    y = np.bincount(y_true[b == i], minlength=n_classes)
    ece += np.abs(f - y).sum() / n
  return ece


## Logging helpers


def evaluate(y_true,
             y_preds,
             groups=None,
             n_classes: Optional[int] = None,
             n_groups: Optional[int] = None,
             return_std_err: bool = False,
             n_resamples: int = 1000,
             random_state: Optional[int] = None):
  n_classes_ = n_classes
  n_classes = n_classes_ or y_true.max() + 1

  if groups is not None and n_groups is None:
    n_groups = groups.max() + 1

  metrics = {}

  # performance metrics
  metrics['accuracy'] = accuracy(y_true, y_preds, return_std_err=return_std_err)
  pr_mode = ['binary'] if n_classes == 2 else ['micro', 'macro']
  for mode in pr_mode:
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_preds, average=mode, zero_division=0)
    metrics[f'precision_{mode}'] = (precision,)
    metrics[f'recall_{mode}'] = (recall,)
    metrics[f'f1_{mode}'] = (f1,)
  if n_classes == 2:
    metrics['fpr_binary'] = ((y_preds[y_true == 0] == 1).mean(),)

  # fairness metrics
  if groups is not None:
    kwargs = dict(
        n_classes=n_classes,
        n_groups=n_groups,
        return_std_err=return_std_err,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    metrics['sp_disparity'] = sp_disparity(y_preds, groups, **kwargs)
    metrics['sp_disparity_weighted'] = sp_disparity(y_preds,
                                                    groups,
                                                    weighted_sum=True,
                                                    ord=1,
                                                    **kwargs)
    metrics['tpr_disparity'] = tpr_disparity(y_true, y_preds, groups, **kwargs)
    metrics['tpr_disparity_weighted'] = tpr_disparity(y_true,
                                                      y_preds,
                                                      groups,
                                                      weighted_sum=True,
                                                      ord=1,
                                                      **kwargs)
    if n_classes > 2:
      if n_classes == 2 and n_classes_ is None:
        warnings.warn(
            "n_classes is not provided and inferred to be 2. Computing binary TPR disparity."
        )
      metrics['tpr_disparity_rms'] = tpr_disparity(y_true,
                                                   y_preds,
                                                   groups,
                                                   ord=2,
                                                   **kwargs)
    if n_classes == 2:
      metrics['fpr_disparity'] = fpr_disparity(y_true, y_preds, groups,
                                               **kwargs)
      metrics['fpr_disparity_weighted'] = fpr_disparity(y_true,
                                                        y_preds,
                                                        groups,
                                                        weighted_sum=True,
                                                        ord=1,
                                                        **kwargs)
    metrics['eo_disparity'] = eo_disparity(y_true, y_preds, groups, **kwargs)
    metrics['eo_disparity_weighted'] = eo_disparity(y_true,
                                                    y_preds,
                                                    groups,
                                                    weighted_sum=True,
                                                    ord=1,
                                                    **kwargs)
  return metrics


class MetricLogger:

  def __init__(self,
               n_classes: int,
               n_groups: Optional[int] = None,
               return_std_err: bool = False,
               n_resamples: int = 1000,
               random_state: Optional[int] = None):
    self.n_classes = n_classes
    self.n_groups = n_groups
    self.return_std_err = return_std_err
    self.n_resamples = n_resamples
    self.random_state = random_state
    self.n_entries = 0
    self.all_metrics = {}

  def log(self, metrics):
    for name, value in metrics.items():
      k = (name, 'mean')
      if k not in self.all_metrics:
        self.all_metrics[k] = [None] * self.n_entries
      self.all_metrics[k].append(value[0])
      if len(value) > 1:
        k = (name, 'std')
        if k not in self.all_metrics:
          self.all_metrics[k] = [None] * self.n_entries
        self.all_metrics[k].append(value[1])
    # Fill in missing entries with None for consistency
    for name in set(k[0] for k in self.all_metrics.keys()) - set(
        metrics.keys()):
      for k in [(name, 'mean'), (name, 'std')]:
        if k in self.all_metrics:
          self.all_metrics[k].append(None)
    self.n_entries += 1

  def log_evaluate(self, y_true, y_preds, groups=None, **kwargs):
    metrics = {
        k: v if isinstance(v, tuple) else (v,) for k, v in kwargs.items()
    }
    metrics.update(
        evaluate(y_true,
                 y_preds,
                 groups,
                 n_classes=self.n_classes,
                 n_groups=self.n_groups,
                 return_std_err=self.return_std_err,
                 n_resamples=self.n_resamples,
                 random_state=self.random_state))
    self.log(metrics)
    return metrics

  def plot(self, fairness_criteria, performance_metric='accuracy', ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1)
    df_metrics = self.df
    y = df_metrics[performance_metric]['mean'].values
    x = df_metrics[fairness_criteria]['mean'].values
    yerr = (None if 'std' not in df_metrics[performance_metric] else
            df_metrics[performance_metric]['std'].values)
    xerr = (None if 'std' not in df_metrics[fairness_criteria] else
            df_metrics[fairness_criteria]['std'].values)
    markers, caps, bars = ax.errorbar(y=y, x=x, yerr=yerr, xerr=xerr, fmt='o')
    for bar in bars:
      bar.set_alpha(0.5)
    ax.set_ylabel(performance_metric)
    ax.set_xlabel(fairness_criteria)
    return ax

  @property
  def df(self):
    return pd.DataFrame(self.all_metrics)

  def to_csv(self, path: str):
    self.df.to_csv(path)
    # print(f"Metrics saved to {path}")

  @classmethod
  def from_csv(cls, path: str, n_classes, *args, **kwargs) -> 'MetricLogger':
    logger = cls(n_classes, *args, **kwargs)
    if os.path.exists(path):
      df = pd.read_csv(path, header=[0, 1], index_col=[0])
      logger.n_entries = len(df)
      logger.all_metrics = df.to_dict(orient='list')
    return logger

  @staticmethod
  def stringify(metrics):
    s = ''
    for name, values in metrics.items():
      if len(values) > 1:
        s += f"{name}: {values[0]:.4f} ± {values[1]:.4f}\n"
      elif isinstance(values[0], float):
        s += f"{name}: {values[0]:.4f}\n"
      else:
        s += f"{name}: {values[0]}\n"
    return s.strip()

  def __repr__(self):
    s = "MetricLogger:\n\n"
    for i in range(self.n_entries):
      metrics = {}
      for name in self.all_metrics.keys():
        name = name[0]
        if name in metrics:
          continue
        k_mean = (name, 'mean')
        k_std = (name, 'std')
        mean = self.all_metrics.get(k_mean, [None])[i]
        std = self.all_metrics.get(k_std, [None])[i]
        if mean is not None:
          metrics[name] = (mean,) if std is None else (mean, std)
      s += self.stringify(metrics) + '\n'
    return s.strip()
