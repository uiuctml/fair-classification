from typing import Optional, Literal

import numpy as np
import torch
from torch import nn
from sklearn.base import clone

from .utils import projection_simplex


class MLPClassifier:

  def __init__(self,
               hidden_layer_sizes=(100, 100),
               activation=torch.nn.ReLU(),
               n_classes=None,
               n_epochs=20,
               batch_size=128,
               lr=1e-3,
               gamma=0.8,
               device='cpu',
               random_state=33):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.n_classes = n_classes
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.device = device
    self.random_state = random_state
    self.model = None

  def fit(self, X, y, sample_weight=None):

    if self.n_classes is None:
      self.n_classes = max(y) + 1

    X = torch.as_tensor(X).to(self.device, dtype=torch.float32)
    y = torch.as_tensor(y).to(self.device, dtype=torch.long)

    if sample_weight is None:
      sample_weight = [1] * len(y)
    sample_weight = torch.as_tensor(sample_weight).to(self.device,
                                                      dtype=torch.float32)

    # Initialize layers
    torch.manual_seed(self.random_state)
    layers = []
    hidden_layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes)
    for i in range(1, len(hidden_layer_sizes)):
      layers.append(
          torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
      layers.append(self.activation)
    layers.append(torch.nn.Linear(hidden_layer_sizes[-1], self.n_classes))
    self.model = torch.nn.Sequential(*layers).to(self.device)

    dataloader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y, sample_weight),
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=self.gamma)

    self.model.train()
    for epoch in range(self.n_epochs):
      for x, y, w in dataloader_train:
        optimizer.zero_grad()
        outputs = self.model(x)
        losses = loss_fn(outputs, y)
        loss = (losses * w).mean()
        loss.backward()
        optimizer.step()
      scheduler.step()

    return self

  def predict_proba(self, X):
    self.model.eval()
    X = torch.as_tensor(X).to(self.device, dtype=torch.float32)
    probas = []
    with torch.no_grad():
      for x in torch.utils.data.DataLoader(
          X,
          batch_size=self.batch_size,
          shuffle=False,
      ):
        probas.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)

  def predict(self, X):
    return self.predict_proba(X).argmax(axis=1)


# Implements https://arxiv.org/abs/2107.05719


class CriticDecision:

  def __init__(self, n_actions=2, lr=0.001, max_iter=100, device='cpu'):
    self.n_actions = n_actions
    self.lr = lr
    self.max_iter = max_iter
    self.device = device

  def fit(self, X, y):
    self.n_classes_ = X.shape[1]
    self.W_ = nn.Parameter(torch.randn((self.n_classes_, self.n_actions)))

    X = torch.as_tensor(X).to(self.device)  # probas P
    y = torch.as_tensor(y, dtype=torch.long)  # labels Y
    diff = (torch.eye(self.n_classes_)[y].to(self.device) - X)  # Y - P
    optimizer = torch.optim.Adam([self.W_], lr=self.lr)

    def closure():
      optimizer.zero_grad()
      probas_action = self.forward_(X)  # action probas B
      R = (probas_action[..., None] * diff[:, None, :]).mean(dim=0)
      loss = (R**2).sum()
      (-loss).backward()
      return loss

    for _ in range(self.max_iter):
      loss = closure()
      optimizer.step()
    self.score_ = loss.item()
    return self

  def forward_(self, X):
    return torch.nn.functional.softmax(X @ self.W_, dim=-1)

  def predict_proba(self, X):
    X = torch.as_tensor(X).to(self.device)
    with torch.no_grad():
      return self.forward_(X).detach().cpu().numpy()

  def predict(self, X):
    return self.predict_proba(X).argmax(axis=-1)


class DecisionCalibrator:

  def __init__(self,
               n_actions=2,
               critic=None,
               max_iter=10,
               projection: Optional[Literal['simplex', 'clip']] = 'simplex'):
    self.n_actions = n_actions
    self.critic = critic
    self.max_iter = max_iter
    self.projection = projection
    self.predict_fns_ = []
    self.adjustments_ = []
    self.scores_ = []

  def fit(self, X, y):
    if self.critic is None:
      self.critic = CriticDecision(n_actions=self.n_actions)
    for _ in range(self.max_iter):
      predictor = clone(self.critic).fit(X, y)
      X = self.add_(X, y, predictor.predict_proba)
    return self

  def add_(self, X, y, predict_fn, probas_action=None, **kwargs):
    n_samples, n_classes = X.shape  # shape = [N, C]
    if probas_action is None:
      probas_action = predict_fn(X, **kwargs)
    diff = np.eye(n_classes)[y] - X  # Y - P
    D = probas_action.T @ probas_action / n_samples
    Di = np.linalg.pinv(D)
    R = np.mean(probas_action[..., None] * diff[:, None, :], axis=0)
    adjustment = R.T @ Di
    self.adjustments_.append(adjustment)
    self.predict_fns_.append(predict_fn)
    self.scores_.append((R**2).sum())
    return self.apply_(X, adjustment, probas_action)

  def apply_(self, X, adjustment, probas_action):
    X = X + (adjustment @ probas_action[..., None]).squeeze(-1)
    if self.projection == 'simplex':
      X = projection_simplex(X, axis=1)
    elif self.projection == 'clip':
      X = np.clip(X, 0, 1)
    return X

  def predict_proba(self, X, **kwargs):
    for predict_fn, adjustment in zip(self.predict_fns_, self.adjustments_):
      X = self.apply_(X, adjustment, predict_fn(X, **kwargs))
    return X
