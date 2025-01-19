from copy import deepcopy
from functools import partial
import traceback

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm

from joblib import Parallel, delayed
# from tqdm.contrib.concurrent import process_map

from models import BinningCalibrator
import postprocess


def error_rate(y_true, y_preds, groups=None, w=None, n_groups=None):
  """Compute group-weighted error rate."""
  if groups is None or w is None:
    return np.mean(y_true != y_preds)
  else:
    if n_groups is None:
      group_names, groups = np.unique(groups, return_inverse=True)
      n_groups = len(group_names)
    return sum([
        w[a] * np.mean(y_true[groups == a] != y_preds[groups == a])
        for a in range(n_groups)
    ])


def delta_sp(y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of statistical parity."""
  pred_counts = np.array([
      np.bincount(y_preds[groups == a], minlength=n_classes)
      for a in range(n_groups)
  ])
  output_dists = pred_counts / np.sum(pred_counts, axis=1, keepdims=True)
  diffs = np.linalg.norm(output_dists[:, None, :] - output_dists[None, :, :],
                         ord=ord,
                         axis=2)
  return np.max(diffs)


def confusion_matrix(y_true, y_preds, groups, n_classes, n_groups):
  """Compute group-wise confusion matrices (conditioned on y_true)."""
  return np.array([
      sklearn.metrics.confusion_matrix(y_true[groups == a],
                                       y_preds[groups == a],
                                       labels=np.arange(n_classes),
                                       normalize='true')
      for a in range(n_groups)
  ])


def delta_eo(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of equalized odds."""
  conf_mtxs = confusion_matrix(
      y_true,
      y_preds,
      groups,
      n_classes,
      n_groups,
  ).reshape(n_groups, -1)  # shape = (n_groups, n_classes**2)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(conf_mtxs[:, None, :] - conf_mtxs[None, :, :],
                           ord=ord,
                           axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


def delta_eopp(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """
  Compute violation of (binary or multi-class) equalized opportunity (depending
  on `n_classes`).
  """
  conf_mtxs = confusion_matrix(
      y_true, y_preds, groups, n_classes,
      n_groups)  # shape = (n_groups, n_classes, n_classes)
  tprs = np.array([np.diag(conf_mtx) for conf_mtx in conf_mtxs
                  ])  # shape = (n_groups, n_classes)
  if n_classes == 2:
    tprs = tprs[:, 1].reshape(-1, 1)  # shape = (n_groups, 1)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(tprs[:, None, :] - tprs[None, :, :], ord=ord, axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


def calibration_error(probas, labels, n_bins=100, seed=0):
  """Computes binned expected calibration error, with bins selected by k-means.
  """
  calib = BinningCalibrator(n_bins=n_bins,
                            random_state=seed).fit(probas, labels)
  # bins = calib.binning_fn_(probas)
  # bin_to_proba = {b: probas[bins == b].mean(axis=0) for b in np.unique(bins)}
  # probas_binned = np.array([bin_to_proba[b] for b in bins])
  p = np.mean(probas, axis=0)
  probas_cal = calib.predict_proba(probas)
  p_cal = np.mean(probas_cal, axis=0)
  return np.max(np.mean(np.abs(probas / p - probas_cal / p_cal), axis=0))


# Define some utility functions


def postprocess_and_evaluate(
    alphas,
    seeds,
    criterion,
    metrics,
    n_test,
    n_classes,
    n_groups,
    labels,
    groups,
    p_y_x=None,
    p_a_x=None,
    p_ay_x=None,
    calibrator_y_factory=None,
    calibrator_a_factory=None,
    calibrator_ay_factory=None,
    max_workers=1,
    #  postproc_kwargs=None,
    return_vals=False,
    print_code=False):

  ## This wrapper is for our algorithm defined in postprocess.PostProcessor

  # if postproc_kwargs is None:
  #   postproc_kwargs = {}

  if print_code:
    if p_ay_x is not None:
      print(
          f'''Code for post-processing a single model (with precomputed probas):

    postprocessor = postprocess.PostProcessor(
        n_classes,
        n_groups,
        pred_ay_fn=lambda x: x,  # dummy pred_fn
        criterion='{criterion}',
        alpha=alpha,
        seed=seed,
    )
    postprocessor.fit(p_ay_x_postproc)
    preds = postprocessor.predict(p_ay_x_test)''')
    else:
      print(
          f'''Code for post-processing a single model (with precomputed probas):

    postprocessor = postprocess.PostProcessor(
        n_classes,
        n_groups,
        pred_a_fn=lambda x: x[0],  # dummy pred_fns
        pred_y_fn=lambda x: x[1],
        criterion='{criterion}',
        alpha=alpha,
        seed=seed,
    )
    postprocessor.fit([p_a_x_postproc, p_y_x_postproc])
    preds = postprocessor.predict((p_a_x_test, p_y_x_test))''')

  kwargs = {
      'postprocessor_factory': None,
      'metrics': metrics,
      'labels_te': None,
      'groups_te': None,
      'p_y_x_pp': None,
      'p_y_x_te': None,
      'p_a_x_pp': None,
      'p_a_x_te': None,
      'p_ay_x_pp': None,
      'p_ay_x_te': None,
  }

  pp_kwargs = []

  for seed in seeds:

    # Split the remaining data into post-processing and test data
    n = len(labels)
    idx_te = np.random.default_rng(seed).choice(np.arange(n),
                                                size=n_test,
                                                replace=False)
    idx_pp = np.setdiff1d(np.arange(n), idx_te)

    labels_pp = labels[idx_pp]
    groups_pp = groups[idx_pp]
    labels_te = labels[idx_te]
    groups_te = groups[idx_te]

    kwargs['labels_te'] = labels_te
    kwargs['groups_te'] = groups_te

    def calibrate(calibrator_factory, probas_pp, targets_pp, probas_te):
      calib = calibrator_factory()
      calib.fit(probas_pp.reshape(len(probas_pp), -1), targets_pp)
      probas_pp = calib.predict_proba(probas_pp.reshape(
          len(probas_pp), -1)).reshape(probas_pp.shape)
      probas_te = calib.predict_proba(probas_te.reshape(
          len(probas_te), -1)).reshape(probas_te.shape)
      return probas_pp, probas_te

    if p_y_x is not None:
      p_y_x_pp = p_y_x[idx_pp]
      p_y_x_te = p_y_x[idx_te]
      if calibrator_y_factory is not None:
        p_y_x_pp, p_y_x_te = calibrate(calibrator_y_factory, p_y_x_pp,
                                       labels_pp, p_y_x_te)
      kwargs['p_y_x_pp'] = p_y_x_pp
      kwargs['p_y_x_te'] = p_y_x_te
    if p_a_x is not None:
      p_a_x_pp = p_a_x[idx_pp]
      p_a_x_te = p_a_x[idx_te]
      if calibrator_a_factory is not None:
        p_a_x_pp, p_a_x_te = calibrate(calibrator_a_factory, p_a_x_pp,
                                       groups_pp, p_a_x_te)
      kwargs['p_a_x_pp'] = p_a_x_pp
      kwargs['p_a_x_te'] = p_a_x_te
    if p_ay_x is not None:
      p_ay_x_pp = p_ay_x[idx_pp]
      p_ay_x_te = p_ay_x[idx_te]
      if calibrator_ay_factory is not None:
        p_ay_x_pp, p_ay_x_te = calibrate(calibrator_ay_factory, p_ay_x_pp,
                                         groups_pp * n_classes + labels_pp,
                                         p_ay_x_te)
      kwargs['p_ay_x_pp'] = p_ay_x_pp
      kwargs['p_ay_x_te'] = p_ay_x_te

    for alpha in alphas:
      kwargs['postprocessor_factory'] = partial(
          postprocess.PostProcessor,
          alpha=alpha,
          seed=seed,
          n_classes=n_classes,
          n_groups=n_groups,
          criterion=criterion,
          # **postproc_kwargs,
      )

      pp_kwargs.append(deepcopy(kwargs))

  if max_workers == 1:
    res = []
    for a in pp_kwargs:
      res.append(postprocess_and_evaluate_(a))
      # print(res[-1])  # to monitor progress
  else:
    res = Parallel(n_jobs=max_workers)(
        delayed(postprocess_and_evaluate_)(pp_kwargs[i])
        for i in tqdm.tqdm(range(len(pp_kwargs))))

    ## process_map does not work with sklearn
    # res = process_map(
    #     postprocess_and_evaluate_,
    #     pp_kwargs,
    #     max_workers=max_workers,
    # )

  # each r in res is (alpha, seed, metrics, postprocessor)
  ret = pd.DataFrame([{
      'alpha': alpha,
      **result
  } for alpha, _, result, _ in res if result is not None])
  ret = ret.groupby('alpha').agg(['mean', np.std]).sort_index(ascending=False)
  if return_vals:
    return ret, res
  return ret


def dict_get_key(d, k):
  return d[k]


def postprocess_and_evaluate_(kwargs):

  postprocessor_factory = kwargs['postprocessor_factory']
  metrics = kwargs['metrics']
  labels_te = kwargs['labels_te']
  groups_te = kwargs['groups_te']
  p_a_x_pp = kwargs['p_a_x_pp']
  p_a_x_te = kwargs['p_a_x_te']
  p_y_x_pp = kwargs['p_y_x_pp']
  p_y_x_te = kwargs['p_y_x_te']
  p_ay_x_pp = kwargs['p_ay_x_pp']
  p_ay_x_te = kwargs['p_ay_x_te']

  postprocessor = postprocessor_factory()
  n_classes = postprocessor.n_classes
  n_groups = postprocessor.n_groups
  alpha = postprocessor.alpha
  seed = postprocessor.seed

  try:
    # Post-process the predicted probabilities
    if postprocessor.alpha == np.inf:
      if p_y_x_te is None:
        p_y_x_te = p_ay_x_te.sum(axis=1)
      preds_te = p_y_x_te.argmax(axis=1)
    else:
      postprocessor.fit(None, p_a_x_pp, p_y_x_pp, p_ay_x_pp)
      # Evaluate the post-processed model
      preds_te = postprocessor.predict(None, p_a_x_te, p_y_x_te, p_ay_x_te)
  except Exception:
    print(
        f"Post-processing failed with alpha={alpha} and seed={seed}:\n{traceback.format_exc()}",
        flush=True)
    return alpha, seed, None, None

  return alpha, seed, evaluate(labels_te,
                               preds_te,
                               groups_te,
                               n_groups=n_groups,
                               n_classes=n_classes,
                               metrics=metrics), postprocessor


def evaluate(test_labels,
             test_preds,
             test_groups,
             n_groups=2,
             n_classes=2,
             metrics=[]):
  result = {}
  for metric in metrics:
    if metric == 'accuracy':
      result[metric] = 1 - error_rate(
          test_labels,
          test_preds,
          test_groups,
          n_groups=n_groups,
      )
    elif metric.startswith('delta_sp'):
      result[metric] = delta_sp(
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (np.sqrt(n_classes) if metric.endswith('rms') else 1)
    elif metric.startswith('delta_eopp'):
      result[metric] = delta_eopp(
          test_labels,
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (np.sqrt(n_classes) if
           (metric.endswith('rms') and n_classes > 2) else 1)
    elif metric.startswith('delta_eo'):
      result[metric] = delta_eo(
          test_labels,
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (n_classes if metric.endswith('rms') else 1)
    elif metric.startswith('dist'):
      label = int(metric.split('_')[-1])
      result[metric] = (test_preds == label).mean()
  return result


def plot_results(ax, df, x_col, y_col, label=None, **kwargs):
  if 'fmt' not in kwargs:
    kwargs['fmt'] = '-'
  markers, caps, bars = ax.errorbar(
      df[x_col]['mean'].values,
      df[y_col]['mean'].values,
      xerr=df[x_col]['std'].values,
      yerr=df[y_col]['std'].values,
      lw=2,
      label=label,
      **kwargs,
  )
  for b in bars:
    b.set_alpha(0.4)
  ax.set_xlabel(x_col)
  ax.set_ylabel(y_col)
  ax.grid(True, which="both", zorder=0)
