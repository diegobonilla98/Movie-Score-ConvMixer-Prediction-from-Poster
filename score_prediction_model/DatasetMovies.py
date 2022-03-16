import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np


class DatasetMovies(Dataset):
    def __init__(self, model_type="imagenet"):
        self.model_type = model_type
        assert model_type in ["imagenet", "-1 to 1", "0 to 1"]
        self.transform = transforms.Compose([self.ToTensor()])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.imagenet_transform = transforms.Compose([normalize])
        self.df = pd.read_csv('/media/bonilla/My Book/MoviePosterScore/movies_info.csv')

    @staticmethod
    class ToTensor(object):
        def __call__(self, sample):
            image, score = sample['image'], sample['rating']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'rating': torch.from_numpy(score)}

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        image = cv2.imread('/media/bonilla/My Book/MoviePosterScore/movie_posters/' + x['image_id'] + '.jpg')
        image = cv2.resize(image, (224, 224))
        image = image[:, :, ::-1].astype('float32') / 255.
        if self.model_type != "imagenet":
            if self.model_type == "-1 to 1":
                image = (image * 2.) - 1.
        score = np.array(x['score'] / 10., dtype='float32')
        sample = {'image': image, 'rating': score}
        sample = self.transform(sample)
        if self.model_type == "imagenet":
            sample['image'] = self.imagenet_transform(sample['image'])
        return sample


if __name__ == '__main__':
    ds = DatasetMovies()
    a = ds[0]
