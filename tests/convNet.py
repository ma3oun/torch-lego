from torch_lego import ConvModule, LinearModule
from typing import Union, Tuple, List

import argparse
import ruamel.yaml as yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Union[optim.Adam, optim.SGD],
    epoch: int,
):

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.type(torch.long).to(device)
        optimizer.zero_grad()
        logits = model(data)
        xEntropyLoss = loss_fn(logits, target)
        xEntropyLoss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\n"
                f"\tLoss: {xEntropyLoss.item():.4f}"
            )
    return


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
) -> Tuple[float, int]:

    model.eval()
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.type(torch.long).to(device)
            logits = model(data)

            xEntropyLoss = loss_fn(logits, target)
            predictedLabel = logits.argmax(dim=1, keepdim=True)
            correct += predictedLabel.eq(target.view_as(predictedLabel)).sum()

    testLoss = xEntropyLoss / len(test_loader.dataset)
    accuracyPercent = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {testLoss.item():.4f}\n"
        f"Accuracy: {correct.item()}/{len(test_loader.dataset)} "
        f"({accuracyPercent.item():.0f}%)"
    )

    return accuracyPercent.item(), correct.item()


def assembleMainModule(architecture: dict) -> nn.Module:
    class MainModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ConvModule(architecture["encoder"])
            self.classifier = LinearModule(architecture["classifier"])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.encoder(x)
            y = nn.Flatten()(y)
            y = self.classifier(y)
            return y

    return MainModule()


def getDatasets(batchSize: int) -> Tuple:
    tfms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainDataset = datasets.MNIST("../data", train=True, download=True, transform=tfms)
    testDataset = datasets.MNIST("../data", train=False, transform=tfms)

    train_loader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        testDataset,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
    )
    return train_loader, test_loader


def run(runParams: dict) -> float:
    if not runParams["paramsFile"] is None:
        with open(runParams["paramsFile"], "r") as f:
            params = yaml.safe_load(f)
            params["device"] = runParams["device"]
            params["subDeviceIdx"] = runParams["subDeviceIdx"]
    else:
        params = runParams

    if params["device"] != "cuda":
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()

    if params["subDeviceIdx"] is None:
        subDeviceIdx = 0
    else:
        subDeviceIdx = params["subDeviceIdx"]

    device = torch.device("cuda:{}".format(subDeviceIdx) if use_cuda else "cpu")
    seed = params["training"]["seed"]
    if seed is None:
        seed = np.random.randint(10000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    batchSize = params["training"]["batchSize"]
    epochs = params["training"]["epochs"]
    learningRate = params["training"]["learningRate"]

    net = assembleMainModule(params["architecture"]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learningRate)

    currentAcc = 0
    currentCorrect = 0
    train_loader, test_loader = getDatasets(batchSize)
    for epoch in range(1, epochs + 1):
        print(f"\n\t==== TRAIN (epoch: {epoch})====\n")
        train(net, device, train_loader, optimizer, epoch)
        print("\n\t==== TEST ====\n")
        currentAcc, currentCorrect = test(net, device, test_loader)

    return currentAcc, currentCorrect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection trainer")
    parser.add_argument(
        "configuration", type=str, default=None, help="Configuration file"
    )
    parser.add_argument(
        "device", type=str, help="Device to use", default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument("-d", type=int, help="Cuda device index", default=0)

    args = parser.parse_args()
    paramsFile = args.configuration
    devParam = args.device
    subDeviceIdx = args.d

    run(
        {
            "paramsFile": paramsFile,
            "device": devParam,
            "subDeviceIdx": subDeviceIdx,
        }
    )
