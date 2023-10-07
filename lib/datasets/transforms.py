import numpy as np
import random
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import Image
# from lib.datasets.transforms import make_transforms

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):

    def __call__(self, img, kpts, mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, kpts, mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask

class SaltAndPepperNoise(object):
    def __init__(self, probability=0.95, amount=0.02):
        self.probability = probability
        self.amount = amount

    def __call__(self, image, kpts, mask):
        noisy_image = np.copy(image)
        height, width, _ = noisy_image.shape
        salt_pepper_pixels = np.random.choice([0, 255], size=(height, width), p=[1 - self.probability, self.probability])
        salt_pepper_mask = np.random.random(size=(height, width)) < self.amount

        # Add salt and pepper noise to each channel of the image
        for channel in range(3):
            noisy_image[:, :, channel] = np.where(salt_pepper_mask, salt_pepper_pixels, noisy_image[:, :, channel])

        return noisy_image, kpts, mask


class SparkleNoise(object):
    def __init__(self, probability=0.1, intensity_range=(150, 180)):
        self.probability = probability
        self.intensity_range = intensity_range

    def __call__(self, image, kpts, mask):
        noisy_image = np.copy(image)

        # Generate random bright pixels
        height, width, _ = noisy_image.shape
        sparkle_pixels = np.random.random(size=(height, width)) < self.probability
        sparkle_pixels = sparkle_pixels[..., np.newaxis]
        sparkle_pixels = np.tile(sparkle_pixels, (1, 1, 3))
        sparkle_pixels = sparkle_pixels.astype(np.uint8)

        # Set random intensity for the bright pixels
        intensity = np.random.randint(*self.intensity_range, size=(height, width, 3), dtype=np.uint8)
        
        # Add sparkle noise to the image
        noisy_image = np.where(sparkle_pixels, intensity, noisy_image)

        return noisy_image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask
    

def make_transforms(cfg, is_train):
    if is_train is True:
        transform = Compose(
            [
                ColorJitter(0.2, 0.01, 0.1, 0.1),
                SparkleNoise(probability=0.3, intensity_range=(50, 150)),
                RandomBlur(0.9),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform