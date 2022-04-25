import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import numpy as np
import matplotlib

matplotlib.use('TKAgg')


class DL_2DNR(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
                self.ToNormTensor()
            ])
        self.image = cv2.imread('image.png')[:, :, ::-1]
        self.image_flattened = self.image.copy().reshape((self.image.shape[0] * self.image.shape[1], 3))

    @staticmethod
    class ToNormTensor(object):
        def __call__(self, sample):
            input_coordinates, output_quantity = sample['input_coordinates'], sample['output_quantity']
            input_coordinates = torch.from_numpy(input_coordinates)
            output_quantity = torch.from_numpy(output_quantity)
            return {'input_coordinates': input_coordinates, 'output_quantity': output_quantity}

    def __len__(self):
        return np.prod(self.image.shape[:2])

    def __getitem__(self, i):
        x = (i % self.image.shape[1]) / self.image.shape[1]
        y = (i // self.image.shape[1]) / self.image.shape[1]

        input_coords = np.expand_dims(np.array([x, y]), axis=0)

        output_q = self.image_flattened[i] / 255.
        # output_q[0] = (output_q[0] - 0.485) / 0.229
        # output_q[1] = (output_q[1] - 0.456) / 0.224
        # output_q[2] = (output_q[2] - 0.406) / 0.225
        output_q = np.expand_dims(output_q, axis=0)

        sample = {'input_coordinates': input_coords.astype(np.float32), 'output_quantity': output_q.astype(np.float32)}
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    dl = DL_2DNR()
    a = dl[0]
    print()
