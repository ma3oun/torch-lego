from typing import Union
import torch
import torch.nn as nn
from torch_lego.lego_base import LegoModule, genActivation


class ConvModuleBase(LegoModule):
    def __init__(self, architecture: Union[list, dict, str], transpose: bool) -> None:
        super().__init__(architecture)
        self.transpose = transpose
        if self.transpose:
            layerTypeKey = "tconv"
        else:
            layerTypeKey = "conv"
        for idx, layerParams in enumerate(self.architecture):
            kernelSize = layerParams[layerTypeKey]["kernel_size"]
            if isinstance(kernelSize, list):
                self.dim = len(kernelSize)
            else:
                self.dim = 1
            conv = getattr(nn, f"Conv{self.dim}d")
            tconv = getattr(nn, f"ConvTranspose{self.dim}d")
            bn = getattr(nn, f"BatchNorm{self.dim}d")
            apool = getattr(nn, f"AvgPool{self.dim}d")
            mpool = getattr(nn, f"MaxPool{self.dim}d")
            if self.dim > 1:
                drpt = getattr(nn, f"Dropout{self.dim}d")
            else:
                drpt = nn.Dropout
            if not self.transpose:
                self.__setattr__(f"conv_{idx}", conv(**layerParams[layerTypeKey]))
            else:
                self.__setattr__(f"tconv_{idx}", tconv(**layerParams[layerTypeKey]))
            if "bn" in layerParams.keys():
                self.__setattr__(
                    f"coderBN_{idx}", bn(layerParams[layerTypeKey]["out_channels"])
                )
            else:
                self.__setattr__(f"coderBN_{idx}", nn.Identity())
            if "apool" in layerParams.keys():
                self.__setattr__(f"apool_{idx}", apool(**layerParams["apool"]))
            else:
                self.__setattr__(f"apool_{idx}", nn.Identity())
            if "mpool" in layerParams.keys():
                self.__setattr__(f"mpool_{idx}", mpool(**layerParams["mpool"]))
            else:
                self.__setattr__(f"mpool_{idx}", nn.Identity())
            actType = layerParams["act"]["type"]
            actParams = layerParams["act"]["params"]
            self.__setattr__(f"coderAct_{idx}", genActivation(actType, actParams))
            if "drpt" in layerParams.keys():
                p = layerParams["drpt"]
                self.__setattr__(f"drpt_{idx}", drpt(p=p))
            else:
                self.__setattr__(f"drpt_{idx}", nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dataShape = x.shape
        if self.dim < len(dataShape) - 2:  # remove batch dim and one dimension
            # data dimensionality needs to be reduced
            y = x.reshape(list(dataShape[:2]) + [-1])
        else:
            y = x

        for idx in range(self.nLayers):
            if not self.transpose:
                y = self.__getattr__(f"conv_{idx}")(y)
            else:
                y = self.__getattr__(f"tconv_{idx}")(y)
            y = self.__getattr__(f"apool_{idx}")(y)
            y = self.__getattr__(f"mpool_{idx}")(y)
            y = self.__getattr__(f"coderBN_{idx}")(y)
            y = self.__getattr__(f"coderAct_{idx}")(y)
            y = self.__getattr__(f"drpt_{idx}")(y)
        return y


class ConvModule(ConvModuleBase):
    def __init__(self, architecture: Union[list, dict, str]) -> None:
        super().__init__(architecture, False)


class TransposeConvModule(ConvModuleBase):
    def __init__(self, architecture: Union[list, dict, str]) -> None:
        super().__init__(architecture, True)
