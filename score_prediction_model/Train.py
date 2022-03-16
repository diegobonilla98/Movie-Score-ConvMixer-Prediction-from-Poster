import torch.nn
from model.Model import ConvMixerModel
from BoniDL import utils
from model.DatasetMovies import DatasetMovies
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np


BATCH_SIZE = 16
LEARNING_RATE = 1e-4
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 30

model = ConvMixerModel(huge=False)
print(model)
utils.count_parameters(model)

data_loader = DataLoader(DatasetMovies(model_type="imagenet"), batch_size=BATCH_SIZE, shuffle=True)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
loss_reg = torch.nn.HuberLoss()

if USE_CUDA:
    model = model.cuda()
    loss_reg = loss_reg.cuda()

for p in model.parameters():
    p.requires_grad = True

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_loss = []
    while i < len(data_loader):
        sample = next(data_iter)
        s_image, s_score = sample['image'], sample['rating']

        model.zero_grad()
        if USE_CUDA:
            s_image = s_image.cuda()
            s_score = s_score.cuda()

        s_image_v = Variable(s_image)
        s_score_v = Variable(s_score)

        s_score_output = model(s_image_v)
        err = loss_reg(s_score_output.flatten(), s_score_v.flatten())

        err.backward()
        optimizer.step()

        i += 1
        epoch_loss.append(err.cpu().data.numpy())

    print(f'[Epoch: {epoch}/{N_EPOCHS}, Iter: {i}/{len(data_loader)}], [Error: {np.mean(epoch_loss)}]')
    torch.save(model, f'./convmix_checkpoints/epoch_{epoch}.pth')
