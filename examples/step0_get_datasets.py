import os
import sys
from functools import partial

sys.path.append('..')

import numpy as np

from linearpost import loader


def get_dataset_xz24_tabular(data_dir_base,
                             dataset_name,
                             remove_sensitive_attr=True,
                             seed=None):
  data_dir = os.path.join(data_dir_base, dataset_name)

  if dataset_name == 'adult':
    sensitive_attr = 'sex'
    loader_outputs = loader.adult(
        data_dir,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.3, 0.2, 0.2, 0.3]

  elif dataset_name == 'compas':
    sensitive_attr = 'race'
    loader_outputs = loader.compas(
        data_dir,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.3, 0.2, 0.2, 0.3]

  elif dataset_name == 'acsincome2':
    sensitive_attr = 'SEX'
    data_dir = os.path.join(data_dir_base, 'acsincome')
    loader_outputs = loader.acsincome(
        data_dir,
        n_classes=2,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.5, 0.1, 0.1, 0.3]

  elif dataset_name == 'acsincome5':
    sensitive_attr = 'RAC1P'
    data_dir = os.path.join(data_dir_base, 'acsincome')
    loader_outputs = loader.acsincome(
        data_dir,
        n_classes=5,
        sensitive_attr=sensitive_attr,
    )
    split_sizes = [0.5, 0.1, 0.1, 0.3]

    # Combine RAC1P categories [3, 4, 5], [6, 7], and [8, 9] into new categories
    # 9997, 9998, and 9999 resp., due to small sample size in some groups.
    # This is also consistent with the UCI Adult dataset.
    category_names = loader_outputs['category_names']
    category_names['RAC1P'].update({
        9997: "American Indian or Alaska Native alone",
        9998: "Asian, Native Hawaiian or Other Pacific Islander alone",
        9999: "Other"
    })

    df = loader_outputs['data']
    df['RAC1P'] = df['RAC1P'].astype(df['RAC1P'].to_numpy().dtype)
    df['RAC1P'] = df['RAC1P'].replace([3, 4, 5], 9997)
    df['RAC1P'] = df['RAC1P'].replace([6, 7], 9998)
    df['RAC1P'] = df['RAC1P'].replace([8, 9], 9999)
    df['RAC1P'] = df['RAC1P'].astype('category')

    # Get group labels
    groups = df['RAC1P'].values
    group_names, groups = np.unique(groups, return_inverse=True)
    loader_outputs['groups'] = groups
    loader_outputs['group_names'] = [
        category_names['RAC1P'][n] for n in group_names
    ]

  else:
    raise NotImplementedError

  D = loader.dataset_from_loader_outputs(loader_outputs)
  D.create_splits(
      split_sizes,
      ['pre', 'post', 'val', 'test'],
      shuffle=True,
      seed=seed,
  )

  if remove_sensitive_attr:
    D['X'].drop([sensitive_attr], inplace=True, axis=1)
  print('  - Dataset columns:', ', '.join(D['X'].columns))
  D.preprocess_tabular('X', train_split_name='pre', inplace=True)

  return D


def get_dataset_xz24_textual(data_dir_base, dataset_name, seed=None):
  data_dir = os.path.join(data_dir_base, dataset_name)

  if dataset_name == 'biasbios':
    loader_outputs = loader.biasbios(data_dir)
    split_sizes = [0.5, 0.1, 0.1, 0.3]
  else:
    raise NotImplementedError

  D = loader.dataset_from_loader_outputs(loader_outputs)
  D.create_splits(
      split_sizes,
      ['pre', 'post', 'val', 'test'],
      shuffle=True,
      seed=seed,
  )
  return D


name_to_getter = {
    "xz24_adult_aware":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='adult',
            remove_sensitive_attr=False,
        ),
    "xz24_adult_blind":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='adult',
        ),
    "xz24_compas_aware":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='compas',
            remove_sensitive_attr=False,
        ),
    "xz24_compas_blind":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='compas',
        ),
    "xz24_acsincome2_aware":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='acsincome2',
            remove_sensitive_attr=False,
        ),
    "xz24_acsincome2_blind":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='acsincome2',
        ),
    "xz24_acsincome5_aware":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='acsincome5',
            remove_sensitive_attr=False,
        ),
    "xz24_acsincome5_blind":
        partial(
            get_dataset_xz24_tabular,
            dataset_name='acsincome5',
        ),
    "xz24_biasbios":
        partial(
            get_dataset_xz24_textual,
            dataset_name='biasbios',
        ),
}
