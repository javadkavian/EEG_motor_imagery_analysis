import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        y = 0
        if(self.y[index] == 1):
            y = 1
        return self.X[index].astype(np.float32), y


class MLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape[1], 100)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(10, 5)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(50, 5)
        # self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.relu3(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
    
    
class NNTrainTest:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device
        
    def train(self, num_epochs, train_dataset, batch_size, lr):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.reshape(*self.model.input_shape).to(self.device)
                labels = labels.to(self.device).type(torch.LongTensor)
                outputs = self.model(images)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    self._print_log(epoch, num_epochs, i + 1, len(train_loader), loss.item())
                    
    def test(self, test_dataset, batch_size):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(*self.model.input_shape).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy on test dataset: {100 * correct / total}%')
            
    def _print_log(self, epoch, num_epochs, step, total_steps, loss):
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{total_steps}], Loss: {loss:.4f}')
        
    def predict(self, x):
        x = x.reshape(*self.model.input_shape).to(self.device)
        outputs = self.model(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
