import os

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from data.data_iterator import SimpleISPRSVaihingenDataset
from models.dummynet import DummyNet

seed = 8671
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

n_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Found available device {}'.format(device))

dir_images = os.path.join(os.getenv('HOME'), 'data/crops/train_strong/img/')
dir_segmaps = os.path.join(os.getenv('HOME'), 'data/crops/train_strong/seg/')
dataset = SimpleISPRSVaihingenDataset(dir_images, dir_segmaps, transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = DummyNet()
model.to(device)

mse = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    
    epoch_loss = 0
    for i, data in enumerate(data_loader):
        image = data['image'].to(device)
        segmap_gt = data['segmap'].to(device)
        
        optimizer.zero_grad()
        segmap_pred = model(image)
        loss = mse(segmap_pred, segmap_gt)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print('iter {: >4d}  |  loss {: >2.4f}'.format(i, loss.item()))
    epoch_loss /= i
    print('Epoch {: >3d}  |  Train Loss {: >2.4f}'.format(epoch, epoch_loss))
        
        
    
    