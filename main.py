from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms, models

from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments


def main(args: Arguments.parse.Namespace):
    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(230),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.dataset, transform=transform)

    if executor.name != 'train':
        args.batch = 1

    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    model = executor.init(model, device, args)
    Path(args.dest).mkdir(exist_ok=True, parents=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)

    executor(model, dataset=dataset,
             criterion=criterion, optimizer=optimizer, scheduler=scheduler,     # train args
             device=device, args=args)


if __name__ == '__main__':
    arguments = Arguments()
    seed(arguments.seed)
    main(arguments)
