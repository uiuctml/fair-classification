import numpy as np
import torch


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
