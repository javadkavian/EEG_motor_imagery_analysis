import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        y = 0
        if(self.y is not None and self.y[index] == 1):
            y = 1
        return self.X[index].astype(np.float32), y


class MLP(nn.Module):
    def __init__(self, input_shape, num_classes, batch_size, lr, num_epochs, device):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        
        self.fc1 = nn.Linear(input_shape[1], 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.relu2 = nn.ReLU()
        self.fc5 = nn.Linear(5, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc5(out)
        out = self.softmax(out)
        return out
    
    def fit(self, X, y):
        train_dataset = TorchDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        for epoch in range(self.num_epochs):
            for i, (batch, labels) in enumerate(train_loader):
                batch = batch.reshape(*self.input_shape).to(self.device)
                labels = labels.to(self.device).type(torch.LongTensor)
                outputs = self.forward(batch)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
            if (epoch + 1) % 50 == 0:
                self._print_log(epoch, self.num_epochs, loss.item())
    
        print()
        
    def predict(self, X):
        scores = self.predict_proba(X)
        return np.argmax(scores, axis=1)
        
    def predict_proba(self, X):
        test_loader = DataLoader(TorchDataset(X), batch_size=self.batch_size, shuffle=False)
        scores = np.empty((0, 2))
        with torch.no_grad():
            for batch, labels in test_loader:
                batch = batch.reshape(*self.input_shape).to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(batch)

            scores = np.concatenate([scores, outputs.data])
            
        return scores 
    
    def _print_log(self, epoch, num_epochs, loss):
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

