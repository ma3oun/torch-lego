from torch_lego import ConvModule, TransposeConvModule
from typing import Union, Tuple, List

import argparse
import ruamel.yaml as yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("conv") != -1 or classname.find("tconv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizerGen: Union[optim.Adam, optim.SGD],
    optimizerDisc: Union[optim.Adam, optim.SGD],
    epoch: int,
):
    realLabel = 1.0
    fakeLabel = 0.0
    generator.train()
    discriminator.train()

    loss_fn = nn.BCELoss()
    for batch_idx, data in enumerate(train_loader, 0):

        ## Discriminator
        # Real data
        data = data[0].to(device)
        discriminator.zero_grad()
        batchSize = data.size(0)
        targets = torch.full((batchSize,), realLabel, dtype=torch.float, device=device)
        output = discriminator(data).view(-1)  # flatten
        discLossReal = loss_fn(output, targets)
        discLossReal.backward()
        discErrReal = discLossReal.mean().item()

        noise = torch.randn(batchSize, 100, 1, 1, device=device)
        fakeData = generator(noise)
        targets.fill_(fakeLabel)
        output = discriminator(fakeData.detach()).view(-1)
        discLossFake = loss_fn(output, targets)
        discLossFake.backward()
        discErrFake = discLossFake.mean().item()

        discErr = discErrReal + discErrFake
        optimizerDisc.step()

        ## Generator
        generator.zero_grad()
        output = discriminator(fakeData).view(-1)
        targets.fill_(realLabel)  # we use real labels to maximize cost for generator

        genLoss = loss_fn(output, targets)
        genLoss.backward()
        genErr = genLoss.mean().item()
        optimizerGen.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\n"
                f"\tDiscLoss: {discErr:.4f}\n"
                f"\tGenLoss: {genErr:.4f}"
            )
    return


def test(
    generator: nn.Module,
    epoch: int,
    device: torch.device,
):

    generator.eval()
    with torch.no_grad():
        noise = torch.randn(64, 100, 1, 1, device=device)
        images = generator(noise).detach().cpu()
        visualImages = vutils.make_grid(images, padding=2, normalize=True).numpy()
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imsave(f"epoch_{epoch}.png", np.transpose(visualImages, (1, 2, 0)))

    return


def getDatasets(batchSize: int, imageSize: Tuple[int, int]) -> DataLoader:
    tfms = transforms.Compose(
        [
            transforms.Resize(imageSize),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    datasetPath = f"{os.getcwd()}/data/"
    trainDataset = datasets.CelebA(datasetPath, download=True, transform=tfms)

    train_loader = DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=3
    )
    return train_loader


def run(runParams: dict):
    if not runParams["paramsFile"] is None:
        with open(runParams["paramsFile"], "r") as f:
            params = yaml.safe_load(f)
            params["device"] = runParams["device"]
            params["subDeviceIdx"] = runParams["subDeviceIdx"]
    else:
        params = runParams

    if params["device"] != "cuda":
        use_cuda = False
        nGPUs = 0
    else:
        use_cuda = torch.cuda.is_available()
        nGPUs = torch.cuda.device_count()

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
    beta1 = params["training"]["beta1"]
    inputSize = params["training"]["imageSize"]

    gen = TransposeConvModule(params["architecture"]["generator"]).to(device)
    disc = ConvModule(params["architecture"]["discriminator"]).to(device)

    if nGPUs > 1:
        gen = nn.parallel.DataParallel(gen, list(range(nGPUs)))
        disc = nn.parallel.DataParallel(disc, list(range(nGPUs)))

    gen.apply(weights_init)
    disc.apply(weights_init)

    optimizerGen = optim.Adam(gen.parameters(), lr=learningRate, betas=(beta1, 0.999))
    optimizerDisc = optim.Adam(disc.parameters(), lr=learningRate, betas=(beta1, 0.999))

    train_loader = getDatasets(batchSize, inputSize)
    for epoch in range(1, epochs + 1):
        print(f"\n\t==== TRAIN (epoch: {epoch})====\n")
        train(gen, disc, device, train_loader, optimizerGen, optimizerDisc, epoch)
        print("\n\t==== TEST ====\n")
        test(gen, epoch, device)

    return


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
