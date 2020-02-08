from pathlib import Path

import torch
import torch.optim as optim
from torchvision import datasets

from models.model import Model
from lib.augmentation import Transform
from utils import seed
from utils.executable import Executable
from utils.arguments import Arguments


def main(args: Arguments.parse.Namespace):
    executor = Executable.s[args.command]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Transform(args.size)
    dataset = datasets.ImageFolder(args.dataset, transform=transform)

    model = Model.new(args.backbone, len(dataset.classes), pretrained=True)

    model = executor.init(model, device, args)
    Path(args.dest).mkdir(exist_ok=True, parents=True)

    criterion = model.LOSS()
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
