import json
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data

from utils.arguments import Arguments


def init(model: nn.Module, device: torch.device,
         args: Arguments.parse.Namespace = None) \
        -> nn.Module:
    model = model.to(device)
    model.load_state_dict(torch.load(args.model, map_location=lambda s, l: s))
    model.eval()

    return model


def eval(model: nn.Module, dataset: data.Dataset,
         device: torch.device = None, args: Arguments.parse.Namespace = None, **kwargs) \
        -> None:
    loader = data.DataLoader(dataset, args.batch, num_workers=args.worker,
                             shuffle=False, pin_memory=True)

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True)
    corrects, total = 0, 0

    with tqdm(total=len(dataset)) as tq:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data).item()
            total += args.batch

            tq.set_postfix(acc=corrects / total)
            tq.update(1)

    with open(str(dest.joinpath('results.json')), 'w') as f:
        json.dump({
            'accuracy': float(corrects / total),
            'total': int(total),
        }, f)

    print(f'acc: {corrects / total}')
    print(f'total: {total}')
