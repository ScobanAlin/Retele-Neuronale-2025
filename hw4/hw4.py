import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import pickle
from torchvision import transforms



def get_device():
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print("Using device:", device)



train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    lambda x: x.view(-1)
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    lambda x: x.view(-1)
])

class ExtendedMNIST(Dataset):
    def __init__(self, pkl_path, transform=None, train=True):
        self.train = train
        self.transform = transform

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if train:
            self.images = [x[0].reshape(28, 28).astype("uint8") for x in data]
            self.labels = [int(x[1]) for x in data]
        else:
            self.images = [x[0].reshape(28, 28).astype("uint8") for x in data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if self.train:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label

        return img



train_dataset = ExtendedMNIST("./extended_mnist_train.pkl", transform=train_transforms, train=True)
test_dataset = ExtendedMNIST("./extended_mnist_test.pkl", transform=test_transforms, train=False)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)



class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden1_size)

        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)

        self.layer_3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer_1(x))

        x = F.relu(self.layer_2(x))

        x = self.layer_3(x)
        return x


model = MLP(784, 1024, 512, 10).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


EPOCHS = 10
for epoch in range(EPOCHS):
    tr = train_one_epoch(model, train_loader, criterion, optimizer, device)


print("Training finished!")



@torch.inference_mode()
def predict(model, loader, device):
    preds = []
    for x in loader:
        x = x.to(device)
        out = model(x)
        preds.extend(out.argmax(1).cpu().numpy())
    return preds


preds = predict(model, test_loader, device)

pd.DataFrame({
    "ID": np.arange(len(preds)),
    "target": preds
}).to_csv("submission.csv", index=False)

print("Saved submission.csv")
