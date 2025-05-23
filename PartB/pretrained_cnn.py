# -*- coding: utf-8 -*-
"""Untitled13.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CauD7nfwC07F10mnSJr1XBft6uVpl0VT
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class PretrainedCNN:
    def __init__(self, trainset, valset, model, batch_size=32, freeze_percent=1):
        self.model = model

        if freeze_percent == 1:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            total_layers = sum(1 for _ in model.parameters())
            num_layers_to_freeze = int(freeze_percent * total_layers)
            frozen_layers = 0
            for param in self.model.parameters():
                param.requires_grad = False
                frozen_layers += 1
                if frozen_layers >= num_layers_to_freeze:
                    break

        num_classes = 10  # Set as per iNaturalist subset
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.dataloader_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.dataloader_val = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    def train(self, epochs=10, lr=0.001, weight_decay=0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_accuracy, count = 0, 0
            for inputs, labels in self.dataloader_train:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_accuracy += (torch.argmax(outputs, 1) == labels).float().sum()
                count += len(labels)
            print(f"Epoch {epoch}: Train Accuracy: {(train_accuracy/count)*100:.2f}%")

            self.model.eval()
            val_accuracy, count = 0, 0
            with torch.no_grad():
                for inputs, labels in self.dataloader_val:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    val_accuracy += (torch.argmax(outputs, 1) == labels).float().sum()
                    count += len(labels)
            print(f"Epoch {epoch}: Validation Accuracy: {(val_accuracy/count)*100:.2f}%")

