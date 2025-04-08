'''
For tabular datasets, it is important to "Mark categorical columns" in order for
dataset.Dataset.preprocess_tabular to one-hot encode them.
'''

import csv
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd

import folktables


def adult(data_dir, sensitive_attr='sex'):
  if isinstance(sensitive_attr, str):
    sensitive_attr = [sensitive_attr]

  column_names = [
      "age", "workclass", "fnlwgt", "education", "education-num",
      "marital-status", "occupation", "relationship", "race", "sex",
      "capital-gain", "capital-loss", "hours-per-week", "native-country",
      "class"
  ]

  # Download data
  train_path = f'{data_dir}/adult.data'
  test_path = f'{data_dir}/adult.test'
  if any([not os.path.exists(p) for p in [train_path, test_path]]):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        train_path)
    urllib.request.urlretrieve(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        test_path)

  df_train = pd.read_csv(train_path,
                         names=column_names,
                         sep=r'\s*,\s*',
                         engine='python',
                         na_values='?')
  df_test = pd.read_csv(test_path,
                        names=column_names,
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values='?',
                        skiprows=1)
  df = pd.concat([df_train, df_test])
  df.drop(['fnlwgt'], inplace=True, axis=1)

  # Rename labels '<=50K.' to '<=50K' and '>50K.' to '>50K'
  df.replace('<=50K.', '<=50K', inplace=True)
  df.replace('>50K.', '>50K', inplace=True)

  # Get class labels, and remove them from input data
  labels = df['class']
  df.drop(['class'], inplace=True, axis=1)

  # Get sensitive attributes
  groups = df[sensitive_attr[0]]
  for attribute in sensitive_attr[1:]:
    groups = np.add(np.add(groups, ", "), df[attribute])

  # Encode labels and groups
  label_names, labels = np.unique(labels, return_inverse=True)
  group_names, groups = np.unique(groups, return_inverse=True)

  # Mark categorical columns
  categotical_columns = [
      "workclass", "education", "marital-status", "occupation", "relationship",
      "race", "sex", "native-country"
  ]
  df[categotical_columns] = df[categotical_columns].apply(
      lambda x: x.astype('category'))

  return {
      'data': df,
      'labels': labels,
      'groups': groups,
      'label_names': label_names,
      'group_names': group_names,
      'column_names': column_names,
      # 'category_names': category_names
      'splits': {
          'train': df_train.shape[0],
          'test': df_test.shape[0]
      }
  }


def acsincome(data_dir, n_classes=2, sensitive_attr='SEX'):
  # PINCP: Total person's income (will be binned into n_classes bins)
  target = 'PINCP'
  # Features to keep
  features = [
      'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX',
      'RAC1P'
  ]

  # Download and parse dataset description
  data_path = os.path.join(data_dir, "PUMS_Data_Dictionary_2018.csv")
  if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.csv',
        data_path)

  column_names = {}
  category_names = {}
  with open(data_path, 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
      if row[1] in features:
        if row[0] == 'NAME':
          column_names[row[1]] = row[-1]
        elif row[2] == 'C':
          if row[1] not in category_names:
            category_names[row[1]] = {}
          k = row[-2].lstrip('0')
          if k == '':
            k = '0'
          category_names[row[1]][k] = row[-1]

  # Download data via folktables
  df_raw = folktables.adult_filter(
      folktables.ACSDataSource(
          survey_year='2018',
          horizon='1-Year',
          survey='person',
          root_dir=data_dir,
      ).get_data(download=True))
  df, targets, groups = folktables.BasicProblem(
      features=features,
      target=target,
      group=sensitive_attr,
      postprocess=lambda x: np.nan_to_num(x, nan=-1)).df_to_pandas(df_raw)

  groups = groups.values.flatten()
  group_names, groups = np.unique(groups, return_inverse=True)
  group_names = [category_names[sensitive_attr][str(n)] for n in group_names]

  targets = targets.values.flatten()
  if n_classes == 2:
    # Binarize PINCP into two classes: <=50K and >50K
    label_names = ["<=50K", ">50K"]
    labels = (targets > 50000).astype(int)
  else:
    # Bin PINCP into n_classes bins of roughly equal number of samples
    # (note there is potential test set leakage in computing the bins)
    labels = pd.qcut(
        targets,
        q=n_classes,
        precision=0,
        duplicates='drop',
    )
    label_names = [str(c) for c in labels.categories]
    labels = labels.codes.astype(int)

  # Mark categorical columns
  categotical_columns = [
      'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P'
  ]
  df[categotical_columns] = df[categotical_columns].apply(
      lambda x: x.astype('category'))

  return {
      'data': df,
      'labels': labels,
      'groups': groups,
      'label_names': label_names,
      'group_names': group_names,
      'column_names': column_names,
      'category_names': category_names
  }


def compas(data_dir, sensitive_attr='sex', keep_textual_features=False):
  if isinstance(sensitive_attr, str):
    sensitive_attr = [sensitive_attr]

  data_path = f"{data_dir}/compas-scores-two-years.csv"
  if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        data_path)

  df = pd.read_csv(data_path)

  # select features for analysis
  df = df[[
      'age', 'c_charge_degree', 'c_charge_desc', 'sex', 'race', 'priors_count',
      'days_b_screening_arrest', 'is_recid', 'c_jail_in', 'c_jail_out'
  ]]

  # Drop missing/bad features (following ProPublica's analysis)
  # ix is the index of variables we want to keep.

  # Remove entries with inconsistent arrest information.
  ix = df['days_b_screening_arrest'] <= 30
  ix = (df['days_b_screening_arrest'] >= -30) & ix

  # remove entries entries where compas case could not be found.
  ix = (df['is_recid'] != -1) & ix

  # remove traffic offenses.
  ix = (df['c_charge_degree'] != "O") & ix

  # trim dataset
  df = df.loc[ix, :]

  # create new attribute "length of stay" with total jail time.
  df['length_of_stay'] = (
      pd.to_datetime(df['c_jail_out']) -
      pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

  # drop 'c_jail_in' and 'c_jail_out'
  # drop columns that won't be used
  dropCol = ['c_jail_in', 'c_jail_out', 'days_b_screening_arrest']
  df.drop(dropCol, inplace=True, axis=1)

  # keep only African-American and Caucasian
  df = df.loc[df['race'].isin(['African-American', 'Caucasian']), :]

  # reset index
  df.reset_index(inplace=True, drop=True)

  # Get class labels, and remove them from input data
  labels = df["is_recid"].replace(0, "No").replace(1, "Yes")
  df.drop(["is_recid"], inplace=True, axis=1)

  # Get sensitive attributes
  groups = df[sensitive_attr[0]]
  for attribute in sensitive_attr[1:]:
    groups = np.add(np.add(groups, ", "), df[attribute])

  # Encode labels and groups
  label_names, labels = np.unique(labels, return_inverse=True)
  group_names, groups = np.unique(groups, return_inverse=True)

  if not keep_textual_features:
    df.drop(['c_charge_desc'], inplace=True, axis=1)

  # Mark categorical columns
  categotical_columns = ['c_charge_degree', 'sex', 'race']
  df[categotical_columns] = df[categotical_columns].apply(
      lambda x: x.astype('category'))

  return {
      'data': df,
      'labels': labels,
      'groups': groups,
      'label_names': label_names,
      'group_names': group_names,
      # 'column_names': column_names,
      # 'category_names': category_names
  }


def biasbios(data_dir):
  train_path = f'{data_dir}/train.pickle'
  test_path = f'{data_dir}/test.pickle'
  dev_path = f'{data_dir}/dev.pickle'
  if any(not os.path.exists(p) for p in [train_path, test_path, dev_path]):
    os.makedirs(data_dir, exist_ok=True)
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle',
        train_path)
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle',
        test_path)
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle',
        dev_path)

  bios = []
  titles = []
  genders = []
  splits = {'train': 0, 'test': 0, 'dev': 0}
  for split, path in zip(['train', 'test', 'dev'],
                         [train_path, test_path, dev_path]):
    with open(path, 'rb') as pickle_file:
      for row in pickle.load(pickle_file):
        bios.append(row['hard_text_untokenized'])
        titles.append(row['p'])
        genders.append(row['g'])
        splits[split] += 1

  labels = np.array(titles)
  groups = np.array(genders)

  # Encode labels and groups
  label_names, labels = np.unique(labels, return_inverse=True)
  group_names, groups = np.unique(groups, return_inverse=True)

  return {
      'data': bios,
      'labels': labels,
      'groups': groups,
      'label_names': label_names,
      'group_names': group_names,
      'splits': splits
  }
