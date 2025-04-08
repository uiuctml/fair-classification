from dataclasses import dataclass
from typing import Optional, Sequence, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DataType = pd.DataFrame | np.ndarray | list


@dataclass
class Feature:
  pass


@dataclass
class Categorical(Feature):
  n_categories: int
  category_names: Optional[list[str]] = None


@dataclass
class Array(Feature):
  column_names: Optional[list[str]] = None
  category_names: Optional[list[str]] = None


class Dataset:

  def __init__(self,
               X: Optional[dict[str, DataType]] = None,
               features: Optional[dict[str, Feature]] = None):
    self.X = X or {}
    self.features = features or {}
    self.split_idx = {}

  def __getitem__(self, name):
    return self.X[name]

  def __setitem__(self, name, data):
    self.add_column(name, data)

  def __len__(self):
    if not self.X:
      return 0
    return len(next(iter(self.X.values())))

  def add_column(self,
                 name: str,
                 data: DataType,
                 feature: Optional[Feature] = None) -> None:
    self.X[name] = data
    if feature is None:
      feature = Feature()
    self.features[name] = feature

  def get_split(self, split_name: str) -> 'Dataset':
    X = {}
    idx = self.split_idx[split_name]
    for name, data in self.X.items():
      if isinstance(data, pd.DataFrame):
        X[name] = data.iloc[idx]
      elif isinstance(data, np.ndarray):
        X[name] = data[idx]
      elif isinstance(data, list):
        X[name] = [data[i] for i in idx]
      else:
        raise NotImplementedError
    return Dataset(X=X, features=self.features)

  def get_splits(
      self,
      split_names: Optional[Sequence[str]] = None) -> dict[str, 'Dataset']:
    if split_names is None:
      split_names = list(self.split_idx.keys())
    return {name: self.get_split(name) for name in split_names}

  def create_splits(self,
                    sizes: Sequence[int | float],
                    names: Sequence[str],
                    shuffle: bool = False,
                    seed: Optional[int] = None) -> None:
    if shuffle:
      idx = np.random.RandomState(seed).permutation(len(self))
    else:
      idx = np.arange(len(self))
    if isinstance(sizes[0], float):
      split_idx = np.cumsum([s * len(self) for s in sizes]).astype(int)
    else:
      split_idx = np.cumsum(sizes)
    idxs = np.split(idx, split_idx[:-1])
    self.split_idx = {name: idx for name, idx in zip(names, idxs)}

  @classmethod
  def from_splits(cls, splits: dict[str, 'Dataset']) -> 'Dataset':
    X = {}
    features = next(iter(splits.values())).features
    split_idx = {}
    for split_name, split in splits.items():
      N = sum(len(idx) for idx in split_idx.values())
      split_idx[split_name] = np.arange(N, N + len(split))
      for name in features:
        x = split.X[name]
        if name not in X:
          X[name] = x
        else:
          if isinstance(X[name], pd.DataFrame) and isinstance(x, pd.DataFrame):
            X[name] = pd.concat([X[name], x], axis=0)
          elif isinstance(X[name], np.ndarray) and isinstance(x, np.ndarray):
            X[name] = np.concatenate((X[name], x), axis=0)
          elif isinstance(X[name], list) and isinstance(x, list):
            X[name].extend(x)
          else:
            raise NotImplementedError
    dataset = cls(X=X, features=features)
    dataset.split_idx = split_idx
    return dataset

  @classmethod
  def from_loader_outputs(cls, D) -> 'Dataset':
    X = {'X': D['data'], 'labels': D['labels']}
    features = {
        'X':
            Array(
                column_names=D['column_names'] if 'column_names' in D else None,
                category_names=D['category_names']
                if 'category_names' in D else None),
        'labels':
            Categorical(n_categories=len(D['label_names']),
                        category_names=D['label_names'])
    }
    if 'groups' in D:
      X['groups'] = D['groups']
      features['groups'] = Categorical(n_categories=len(D['group_names']),
                                       category_names=D['group_names'])
    dataset = cls(X=X, features=features)
    if 'splits' in D:
      dataset.create_splits(sizes=list(D['splits'].values()),
                            names=list(D['splits'].keys()))
    return dataset

  def to_dataloader(self,
                    columns: Optional[Sequence[str]] = None,
                    batch_size: int = 1,
                    collate_fn: Optional[Callable] = None,
                    shuffle: bool = False) -> DataLoader:
    if columns is None:
      columns = list(self.X.keys())
    rows = [{
        name: self.X[name][i] for name in columns
    } for i in range(len(self))]
    return DataLoader(rows,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=shuffle)

  def statistics_categorical_joint(self,
                                   name_1: str,
                                   name_2: str,
                                   normalize: bool = False) -> pd.DataFrame:
    f1, f2 = self.features[name_1], self.features[name_2]
    assert isinstance(f1, Categorical) and isinstance(f2, Categorical)
    category_names_1 = (f1.category_names if f1.category_names is not None else
                        np.arange(f1.n_categories).astype(str).tolist())
    category_names_2 = (f2.category_names if f2.category_names is not None else
                        np.arange(f2.n_categories).astype(str).tolist())
    df_stat = pd.DataFrame(
        np.stack([self[name_1], self[name_2]], axis=1),
        columns=[name_1, name_2],
    ).groupby([name_2, name_1]).size().unstack()
    df_stat.rename(index=dict(enumerate(category_names_2)),
                   columns=dict(enumerate(category_names_1)),
                   inplace=True)
    if normalize:
      df_stat /= df_stat.sum().sum()
    return df_stat

  def preprocess_tabular(self, name, train_split_name=None) -> None:
    x = self[name]
    assert isinstance(x, pd.DataFrame)
    self.X[name] = pd.get_dummies(x)

    if train_split_name is None:
      x_train = self[name]
    else:
      x_train = self.get_split(train_split_name)[name]
    scaler = StandardScaler().fit(x_train)
    self.X[name] = scaler.transform(self[name])
    self.features[name] = Array()
