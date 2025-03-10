import numpy as np
import torch
import torch.nn as nn
import sklearn, sklearn.cluster

class MLPClassifier:

  def __init__(self,
               model,
               targets="y",
               hidden_layer_sizes=(100, 100),
               activation=torch.nn.ReLU(),
               n_classes=None,
               n_epochs=20,
               batch_size=128,
               lr=1e-3,
               gamma=0.8,
               device='cpu',
               random_state=33,
               loss_fn=torch.nn.MSELoss()):
    self.targets = targets
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.n_classes = n_classes
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.device = device
    self.random_state = random_state
    self.model = model
    self.loss_fn = loss_fn

  def fit(self, X, y, sample_weight=None):

    if sample_weight is None:
      sample_weight = np.ones(len(y))

    if self.n_classes is None:
      self.n_classes = len(np.unique(y))

    if self.model is None:
      torch.manual_seed(self.random_state)
      layers = []
      hidden_layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes)
      for i in range(1, len(hidden_layer_sizes)):
        layers.append(
            torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        layers.append(self.activation)
      layers.append(torch.nn.Linear(hidden_layer_sizes[-1], self.n_classes))
      self.model = torch.nn.Sequential(*layers).to(self.device)
    #else:
    #  raise ValueError("Refitting is not supported")

    dataloader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.float32).to(self.device),
            torch.tensor(sample_weight, dtype=torch.float32).to(self.device),
        ),
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    if self.targets == "s":
      dataloader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.long).to(self.device),
            torch.tensor(sample_weight, dtype=torch.float32).to(self.device),
        ),
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
      )

    #loss_fn = torch.nn.MSELoss()
    #loss_fn = self.loss_fn()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=self.gamma)

    self.model.train()
    for epoch in range(self.n_epochs):
      for x, y, w in dataloader_train:
        optimizer.zero_grad()
        outputs, __ = self.model(x)
        losses = self.loss_fn(outputs.squeeze(), y)
        #loss = (losses * w).mean()
        #loss.backward()
        losses.backward()
        optimizer.step()
      scheduler.step()

    return self, dataloader_train
  
  def fine_tune(self, X, y):
    optimizer = torch.optim.Adam(
                                 
                                 list(self.model.layer_5.parameters()), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=self.gamma)
    dataloader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.float32).to(self.device),
        ),
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    self.model.train()
    #for epoch in range(self.n_epochs):
    #for epoch in range(100):
    for epoch in range(50):
      for x, y in dataloader_train:
        optimizer.zero_grad()
        outputs, __ = self.model(x)
        losses = self.loss_fn(outputs.squeeze(), y)
        #loss = (losses * w).mean()
        #loss.backward()
        losses.backward()
        optimizer.step()
      scheduler.step()

    return self

  def predict_proba(self, X):
    self.model.eval()
    probas = []
    with torch.no_grad():
      for x in torch.utils.data.DataLoader(
          torch.tensor(X, dtype=torch.float32).to(self.device),
          batch_size=self.batch_size,
          shuffle=False,
      ):
        #probas.append(torch.softmax(self.model(x)[0], dim=1).cpu().numpy())
        probas.append(self.model(x)[0].cpu().numpy())
    return np.concatenate(probas, axis=0)

  def predict(self, X):
    return self.predict_proba(X).argmax(axis=1)

class NNmodel(nn.Module):
  def __init__(self, input_size=9, n_classes=1):
    super(NNmodel, self).__init__()
    """self.layer_1 = nn.Linear(input_size, 10)
    self.activate_1 = nn.ReLU()
    self.dropout_1 = nn.Dropout(p=0.5)
    self.layer_2 = nn.Linear(10, 50)
    self.activate_2 = nn.ReLU()
    self.layer_3 = nn.Linear(50, 50)
    self.activate_3 = nn.ReLU()
    self.bn_3 = nn.BatchNorm1d(num_features=50)
    self.layer_4 = nn.Linear(50, 50)
    self.activate_4 = nn.ReLU()
    self.layer_5 = nn.Linear(50, n_classes)"""
    self.layer_1 = nn.Linear(input_size, 256)
    self.bn_1 = nn.BatchNorm1d(num_features=10)
    self.dropout_1 = nn.Dropout(p=0.5)
    self.activate_1 = nn.ReLU()
    self.layer_2 = nn.Linear(256, 256)
    self.bn_2 = nn.BatchNorm1d(num_features=50)
    self.dropout_2 = nn.Dropout(p=0.5)
    self.activate_2 = nn.ReLU()
    self.layer_3 = nn.Linear(256, 256)
    self.activate_3 = nn.ReLU()
    self.bn_3 = nn.BatchNorm1d(num_features=50)
    self.dropout_3 = nn.Dropout(p=0.5)
    self.layer_4 = nn.Linear(256, 256)
    self.activate_4 = nn.ReLU()
    self.bn_4 = nn.BatchNorm1d(num_features=100)
    self.layer_5 = nn.Linear(256, n_classes)
    #self.layer_5 = nn.Linear(100, 100)
    #self.activate_5 = nn.ReLU()
    #self.bn_5 = nn.BatchNorm1d(num_features=100, momentum=0.05)
    #self.layer_6 = nn.Linear(100, 100)
    #self.layer_6 = nn.Linear(100, n_classes)
    #self.activate_6 = nn.ReLU()
    #self.layer_7 = nn.Linear(100, n_classes)
    #self.layer_7 = nn.Linear(50, 50)
    #self.activate_7 = nn.ReLU()
    #self.layer_8 = nn.Linear(50, n_classes)
    #self.layer_8 = nn.Linear(50, 50)
    #self.activate_8 = nn.ReLU()
    #self.layer_9 = nn.Linear(50, 50)
    #self.activate_9 = nn.ReLU()
    #self.layer_10 = nn.Linear(50, n_classes)
  
  def forward(self, x):
    """out_1 = self.dropout_1(self.activate_1(self.layer_1(x)))
    out_2 = self.activate_2(self.layer_2(out_1))
    out_3 = self.bn_3(self.activate_3(self.layer_3(out_2)))
    out_4 = self.activate_4(self.layer_4(out_3))
    out_5 = self.layer_5(out_4)"""
    #out_1 = self.dropout_1(self.activate_1(self.layer_1(x)))
    out_1 = self.activate_1(self.layer_1(x))
    out_2 = self.activate_2(self.layer_2(out_1))
    out_3 = self.activate_3(self.layer_3(out_2))
    out_4 = self.activate_4(self.layer_4(out_3))
    out_5 = self.layer_5(out_4)
    #out_5 = self.activate_5(self.layer_5(out_4))
    #out_6 = self.activate_6(self.layer_6(out_5))
    #out_7 = self.layer_7(out_6)
    #out_4 = self.activate_4(self.layer_4(out_3))
    #out_5 = self.activate_5(self.layer_5(out_4))
    #out_6 = self.layer_6(out_5)
    #out_6 = self.activate_6(self.layer_6(out_5))
    #out_7 = self.layer_7(out_6)
    #out_4 = self.activate_4(self.layer_4(out_3))
    #out_5 = self.activate_5(self.layer_5(out_4))
    #out_6 = self.activate_6(self.layer_6(out_5))
    #out_7 = self.activate_7(self.layer_7(out_6))
    #out_8 = self.layer_8(out_7)
    #out_8 = self.activate_8(self.layer_8(out_7))
    #out_9 = self.activate_9(self.layer_9(out_8))
    #out_10 = self.layer_10(out_9)
    #return out_5, out_3, out_4
    return out_5, out_3