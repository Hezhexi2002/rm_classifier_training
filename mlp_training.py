# Copyright (c) 2023 ChenJun

import onnx
import os
import torch
import torchvision
from torch import nn


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 28 * 1, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 9)
        )

    def forward(self, x):
        return self.layers(x)


# Init model
model = MLP()
print(model)

# Load data from folder
dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(os.path.dirname(__file__), 'datasets'),
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ]))
print(dataset)

# Show label names
print("classes:\n", dataset.classes)

# Split dataset into train and test (5:1)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=True)

# Train and evaluate
for epoch in range(5):
    for batch, (x, y) in enumerate(train_loader):
        # Forward
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if batch % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}')

    # Evaluate
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%')

# Save as onnx
dummy_input = torch.randn(1, 20, 28, 1)
torch.onnx.export(model, dummy_input, "mlp.onnx")

# Check onnx
onnx_model = onnx.load("mlp.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
