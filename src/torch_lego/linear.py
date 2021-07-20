from typing import Union
import torch
import torch.nn as nn
from torch_lego.lego_base import LegoModule, genActivation


class LinearModule(LegoModule):
    def __init__(
        self, architecture: Union[list, dict, str], in_features: int = None
    ) -> None:
        super().__init__(architecture)
        lastOutputSize = None
        for idx, params in enumerate(self.architecture):
            currentParams = params
            if idx == 0:
                if in_features is None:
                    lastOutputSize = currentParams["in_features"]
                else:
                    lastOutputSize = in_features
                continue
            linearLayer = nn.Linear(lastOutputSize, currentParams["out_features"])
            actType = currentParams["act"]["type"]
            actParams = currentParams["act"]["params"]
            actLayer = genActivation(actType, actParams)
            self.__setattr__(f"dense_{idx-1}", linearLayer)
            self.__setattr__(f"act_{idx-1}", actLayer)
            if "bn" in currentParams.keys():
                self.__setattr__(
                    f"bn_{idx}", nn.BatchNorm1d(currentParams["out_features"])
                )
            else:
                self.__setattr__(f"bn_{idx-1}", nn.Identity())
            if "drpt" in currentParams.keys():
                p = currentParams["drpt"]
                self.__setattr__(f"drpt_{idx-1}", nn.Dropout(p=p))
            else:
                self.__setattr__(f"drpt_{idx-1}", nn.Identity())
            lastOutputSize = currentParams["out_features"]
        self.nLayers -= 1  # remove 1 because of in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        try:
            for idx in range(self.nLayers):
                y = self.__getattr__(f"dense_{idx}")(y)
                y = self.__getattr__(f"bn_{idx}")(y)
                y = self.__getattr__(f"act_{idx}")(y)
                y = self.__getattr__(f"drpt_{idx}")(y)
        except RuntimeError as e:
            print(
                f"Encountered runtime error, input data shape might help:"
                f"\n{x.shape}. Error: {e}"
            )  # for debug
            raise

        return y
