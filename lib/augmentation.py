from typing import Tuple

from torchvision import transforms


class Transform(object):
    def __init__(self, size,
                 mean: Tuple[float, float, float] = None,
                 std: Tuple[float, float, float] = None):

        self.size = size
        mean = mean or (.485, .456, 406)
        std = std or (.229, .224, .225)

        self.transform = transforms.Compose([
            transforms.Resize(int(size * 1.1)),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)
