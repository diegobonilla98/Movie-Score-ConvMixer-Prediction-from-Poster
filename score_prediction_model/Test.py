import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.autograd import Variable


model = torch.load('./convmix_checkpoints/epoch_11.pth').cuda()
model.eval()

# df = pd.read_csv('/media/bonilla/My Book/MoviePosterScore/movies_info.csv')
# idx = df.shape[0] - 1
# x = df.iloc[idx]
# image_org = cv2.imread('/media/bonilla/My Book/MoviePosterScore/movie_posters/' + x['image_id'] + '.jpg')[:, :, ::-1]

for image_name in glob.glob('./test_posters/*'):
    # image_name = './test_posters/greenbook_76.jpg'
    image_org = cv2.imread(image_name)[:, :, ::-1]

    image = cv2.resize(image_org.copy(), (224, 224))
    image = image.astype('float32') / 255.
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)
    # image = (image * 2.) - 1.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)

    s_image = image.cuda()
    s_image_v = Variable(s_image)
    s_score_output = model(s_image_v)

    # plt.title(f"Real: {int(x['score'] * 10.)}, Predicted: {int(s_score_output.cpu().data.numpy().flatten()[0] * 100.)}")
    # plt.title(f"Real: {int(image_name.split(os.sep)[-1].split('.')[0].split('_')[-1])}, Predicted: {int(s_score_output.cpu().data.numpy().flatten()[0] * 100.)}")
    plt.imshow(image_org)
    plt.show()

    print("Real:", int(image_name.split(os.sep)[-1].split('.')[0].split('_')[-1]))
    print("AI:", int(s_score_output.cpu().data.numpy().flatten()[0] * 100.))
    print()
