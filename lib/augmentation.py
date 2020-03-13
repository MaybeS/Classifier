from typing import Tuple

from torchvision import transforms

from utils.beholder import Beholder


class Augmentation(metaclass=Beholder):
    def __init__(self):
        self.training = True
        self.augment = {
            'train': lambda *_: None,
            'test': lambda *_: None,
        }

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, img):
        if isinstance(self.augment, dict):
            augment = self.augment.get('train' if self.training else 'test')
        elif isinstance(self.augment, transforms.Compose):
            augment = self.augment

        return augment(img)


class Transform(Augmentation):
    def __init__(self, size,
                 mean: Tuple[float, float, float] = None,
                 std: Tuple[float, float, float] = None):
        super().__init__()
        self.size = size
        self.mean = mean or (.485, .456, 406)
        self.std = std or (.229, .224, .225)

        self.augment = {
            'train': transforms.Compose([
                transforms.Resize(int(size * 1.1)),
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]), 'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        }
