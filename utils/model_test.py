import torch
import torch.nn as nn
import os

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, model, test_loader_dict, device):
        self.model = model
        self.test_loader_dict = test_loader_dict
        self.keys = list(self.test_loader_dict.keys())
        self.device = device

    def test(self):
        print("----------------------------------")
        sum_acc = 0
        for i in range(len(self.keys)):
            key = self.keys[i]
            test_loader = self.test_loader_dict[key]
            acc = self.test_one_loader(key, test_loader)
            sum_acc += acc
        print("Avg : ", sum_acc / len(self.keys))
        print("----------------------------------")
        return sum_acc / len(self.keys)

    def test_one_loader(self, key, test_loader):
        self.model.eval()
        correct = torch.zeros(1).squeeze().cuda(self.device)
        total = torch.zeros(1).squeeze().cuda(self.device)
        with torch.no_grad():
            for i, (img, label) in enumerate(test_loader):
                img = img.cuda(self.device)
                label = label.cuda(self.device)

                output, _ = self.model.get_logits_feat(img)
                # output = self.model(img)
                prediction = torch.argmax(output, 1)
                correct += (prediction == label).sum().float()
                total += len(label)

        acc = (correct / total).cpu().detach().data.numpy()
        print(key, acc)
        return acc
