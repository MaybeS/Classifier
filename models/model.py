import torch.nn as nn
from torchvision import models

from models.mobilenet.model import mobilenetv3_small
from models.efficientnet.model import EfficientNet
from models.deepcalib.model import Calib
from utils.beholder import Beholder


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Model(nn.Module, metaclass=Beholder):
    LOSS = nn.CrossEntropyLoss
    SCHEDULER = None
    batch_size = 1

    @classmethod
    def new(cls, backbone, class_num, **kwargs):
        model = getattr(models, backbone, None)

        if model is None:
            # model = cls.get(backbone)(**kwargs)
            if backbone == 'mobilenet_v3':
                model = mobilenetv3_small()
            elif backbone == 'calib':
                model = Calib()
            elif backbone.startswith('efficientnet'):
                model = EfficientNet.from_pretrained(backbone, num_classes=class_num)
        else:
            model = model(**kwargs)

            if backbone.startswith('resnet'):
                model.fc = nn.Linear(model.fc.in_features, class_num)
            elif backbone.startswith('mobilenet'):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)
            elif backbone.startswith('vgg'):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_num)

        model.load = model.load_state_dict
        model.LOSS = cls.LOSS

        return model

    @classmethod
    def loss(cls, *args, **kwargs):
        return cls.LOSS(*args, **kwargs)

    def eval(self):
        super(Model, self).eval()
        self.batch_size = 1

    def train(self, mode: bool = True):
        super(Model, self).train(mode)
        self.batch_size = self.batch_size_

        if not mode:
            self.batch_size = 1
