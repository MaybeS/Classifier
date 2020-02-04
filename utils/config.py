import json
from pathlib import Path


class Config:
    def __init__(self, path: str):
        if path is not None:
            try:
                with open(path) as f:
                    for key, value in json.load(f).items():
                        self.update(key, value)
            except (FileNotFoundError, RuntimeError) as e:
                print(f'Configfile {path} is not exists or can not open')

    def update(self, key, value):
        if isinstance(getattr(self, key, None), dict):
            getattr(self, key).update(value)
        else:
            setattr(self, key, value)

    def sync(self, arguments: dict):
        for key, value in arguments.items():
            if key in self.data.keys():
                self.update(key, value)

    @property
    def data(self):
        return {
            attr: getattr(self, attr)
            for attr in filter(lambda attr: not attr.startswith('__') and attr != 'data' and
                                            not callable(getattr(self, attr)), dir(self))
        }
