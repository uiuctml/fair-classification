from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import Tensor

DataType = pd.DataFrame | np.ndarray | Tensor | list


@dataclass(kw_only=True)
class Feature:
  metadata: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class Categorical(Feature):
  n_categories: Optional[int] = -1
  category_names: Optional[list[str]] = None


@dataclass(kw_only=True)
class Array(Feature):
  column_names: Optional[list[str]] = None
  category_names: Optional[list[str]] = None


class Dataset:

  def __init__(self,
               data: Optional[dict[str, DataType]] = None,
               features: Optional[dict[str, Feature]] = None,
               split_idx: Optional[dict[str, Sequence[int]]] = None):
    self.data = data or {}
    self.features = features or {}
    self.split_idx = split_idx or {}

  def __len__(self):
    if not self.data:
      return 0
    return len(next(iter(self.data.values())))

  def create_index(self) -> None:
    self.set_column('_index', np.arange(len(self)))

  @property
  def column_names(self) -> list[str]:
    return list(self.data.keys())

  def items(self):
    return self.data.items()

  def __setitem__(self, column_name, data):
    self.set_column(column_name, data)

  def set_column(self,
                 name: str,
                 data: DataType,
                 feature: Optional[Feature] = None) -> None:
    self.data[name] = data
    if feature is None:
      feature = Feature()
    self.features[name] = feature

  def __getitem__(
      self, column_names: str | Sequence[str]) -> DataType | tuple[DataType]:
    if isinstance(column_names, str):
      return self.data[column_names]
    else:
      return tuple(self.data[name] for name in column_names)

  class ColumnSubsetter:

    def __init__(self, dataset: 'Dataset'):
      self.dataset = dataset

    def __getitem__(self, column_names: str | Sequence[str]) -> 'Dataset':
      if isinstance(column_names, str):
        column_names = [column_names]
      return Dataset(
          data={k: self.dataset[k] for k in column_names},
          features={k: self.dataset.features[k] for k in column_names},
          split_idx=self.dataset.split_idx)

  @property
  def column(self):
    return self.ColumnSubsetter(self)

  class RowSubsetter:

    def __init__(self, dataset: 'Dataset'):
      self.dataset = dataset

    def __getitem__(self, *args, **kwargs):
      # TODO: slicing removes split information, maybe take set intersection?
      data = {}
      for column_name, d in self.dataset.data.items():
        if isinstance(d, pd.DataFrame):
          data[column_name] = d.iloc.__getitem__(*args, **kwargs)
        elif isinstance(d, (np.ndarray, Tensor)):
          data[column_name] = d.__getitem__(*args)
        elif isinstance(d, list):
          if len(args) == 1 and isinstance(args[0], slice):
            data[column_name] = d[args[0]]
          else:
            data[column_name] = [d[i] for i in args[0]]
        else:
          raise NotImplementedError
      dataset = Dataset(data=data, features=self.dataset.features)
      return dataset

  @property
  def iloc(self):
    return self.RowSubsetter(self)

  def sample(self,
             n: int,
             replace: bool = False,
             seed: Optional[int] = None) -> 'Dataset':
    idx = np.random.default_rng(seed).choice(len(self), size=n, replace=replace)
    return self.iloc[idx]

  class Splitter:

    def __init__(self, dataset: 'Dataset'):
      self.dataset = dataset

    def get_split(self, split_name: str) -> 'Dataset':
      dataset = self.dataset.iloc[self.dataset.split_idx[split_name]]
      dataset.split_idx = {}
      return dataset

    def get_splits(self, split_names: Sequence[str]) -> 'Dataset':
      all_idx = []
      new_split_idx = {}
      for split_name in split_names:
        s = sum(len(I) for I in all_idx)
        I = self.dataset.split_idx[split_name]
        t = s + len(I)
        all_idx.append(I)
        new_split_idx[split_name] = np.arange(s, t)
      dataset = self.dataset.iloc[np.concatenate(all_idx)]
      dataset.split_idx = new_split_idx
      return dataset

    def __getitem__(self, split_names: str | Sequence[str]) -> 'Dataset':
      if isinstance(split_names, str):
        return self.get_split(split_names)
      else:
        return self.get_splits(split_names)

  @property
  def split(self):
    return self.Splitter(self)

  def create_splits(self,
                    split_sizes: Sequence[int | float],
                    split_names: Sequence[str],
                    shuffle: bool = False,
                    seed: Optional[int] = None) -> None:
    if shuffle:
      idx = np.random.default_rng(seed).permutation(len(self))
    else:
      idx = np.arange(len(self))
    if isinstance(split_sizes[0], float):
      split_idx = np.cumsum([s * len(self) for s in split_sizes]).astype(int)
    else:
      split_idx = np.cumsum(split_sizes)
    idxs = np.split(idx, split_idx)[:-1]
    self.split_idx = dict(zip(split_names, idxs))

  def to_path(self, path: str) -> None:
    with open(path, 'wb') as f:
      pickle.dump(
          {
              'data': self.data,
              'features': self.features,
              'split_idx': self.split_idx
          }, f)

  @classmethod
  def from_path(cls, path: str) -> 'Dataset':
    with open(path, 'rb') as f:
      d = pickle.load(f)
    return cls(**d)

  def to_dataloader(self,
                    column_names: Optional[Sequence[str]] = None,
                    batch_size: int = 1,
                    collate_fn: Optional[Callable] = None,
                    shuffle: bool = False) -> DataLoader:
    if column_names is None:
      column_names = list(self.data.keys())
    rows = [{n: self.data[n][i] for n in column_names} for i in range(len(self))]
    return DataLoader(rows,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=shuffle)

  def statistics_categorical_joint(self,
                                   column_name_1: str,
                                   column_name_2: str,
                                   normalize: bool = False) -> pd.DataFrame:
    f1, f2 = self.features[column_name_1], self.features[column_name_2]
    assert isinstance(f1, Categorical) and isinstance(f2, Categorical)
    category_names_1 = (f1.category_names if f1.category_names is not None else
                        np.arange(f1.n_categories).astype(str).tolist())
    category_names_2 = (f2.category_names if f2.category_names is not None else
                        np.arange(f2.n_categories).astype(str).tolist())
    df_stat = pd.DataFrame(
        np.stack([self[column_name_1], self[column_name_2]], axis=1),
        columns=[column_name_1, column_name_2],
    ).groupby([column_name_2, column_name_1]).size().unstack()
    df_stat.rename(index=dict(enumerate(category_names_2)),
                   columns=dict(enumerate(category_names_1)),
                   inplace=True)
    if normalize:
      df_stat /= df_stat.sum().sum()
    return df_stat

  def preprocess_tabular(self,
                         column_name,
                         train_split_name=None,
                         inplace=True) -> None:
    x = self[column_name]
    assert isinstance(x, (pd.DataFrame, np.ndarray))
    self.data[column_name] = pd.get_dummies(x)
    if train_split_name is None:
      x_train = self[column_name]
    else:
      x_train = self.split[train_split_name][column_name]
    scaler = StandardScaler().fit(x_train)
    y = scaler.transform(self[column_name])
    if inplace:
      self.data[column_name] = y
      self.features[column_name] = Array()
    return y

  def __repr__(self):
    s = f'Dataset of length {len(self)} containing:\n'
    for column_name, feature in self.features.items():
      if isinstance(feature, Categorical):
        s += f'  - {column_name} ({feature.n_categories} categories)\n'
      else:
        if isinstance(self.data[column_name], pd.DataFrame):
          s += f'  - {column_name} (DataFrame), shape: {self.data[column_name].shape}\n'
        elif isinstance(self.data[column_name], np.ndarray):
          s += f'  - {column_name} (ndarray), shape: {self.data[column_name].shape}\n'
        elif isinstance(self.data[column_name], Tensor):
          s += f'  - {column_name} (tensor), shape: {self.data[column_name].shape}\n'
        elif isinstance(self.data[column_name], list):
          s += f'  - {column_name} (list)\n'
    if self.split_idx:
      s += 'Splits:\n'
      for split_name, idx in self.split_idx.items():
        s += f'  - {split_name}, length: {len(idx)}\n'
    return s.strip()
