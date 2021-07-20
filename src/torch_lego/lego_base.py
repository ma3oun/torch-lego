from typing import Union
import torch.nn as nn
from torch.nn import Identity
import torch.nn.modules.activation as Activations
import ruamel.yaml as yaml


def genActivation(actType: Union[str, None], params: dict = None):
    if actType is None:
        layer = Identity()
    else:
        if params is None:
            layer = getattr(Activations, actType)()
        else:
            layer = getattr(Activations, actType)(**params)
    return layer


class LegoModule(nn.Module):
    def __init__(self, architecture: Union[list, dict, str]) -> None:
        super().__init__()
        if isinstance(architecture, str):
            # load architecture from yaml
            with open(architecture, "r") as f:
                archi = yaml.safe_load(f)
        else:
            archi = architecture
        if isinstance(archi, dict):
            archi = [archi]
        else:
            try:
                assert isinstance(archi, list)
            except AssertionError:
                print("Unknown format for module generation!")
        self.architecture = archi  # type: list
        self.nLayers = len(self.architecture)

        # for future use
        self.inputShape = None  # type: Union[tuple,list]
        self.outputShape = None  # type: Union[tuple,list]
