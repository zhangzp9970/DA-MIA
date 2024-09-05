import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, save_image2

if __name__ == "__main__":
    batch_size = 32
    class_num = 10
    root_dir = "./logZZPMAIN.export"
    feature_pkl = "/path/to/target/model/trained/using/USPS/feature_extractor.pkl"
    cls_pkl = "/path/to/target/model/trained/using/USPS/cls.pkl"
    newfe_pkl = "/path/to/feature/alignment/model/newfe.pkl"
    inv_pkl = "/path/to/reconstruction/model/myinversion.pkl"
    h = 32
    w = 32

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"M2U export newfe dattack",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([Grayscale(num_output_channels=1), Resize((h, w)), ToTensor()])

    mnist_train_ds = USPS(root="./data", train=True, transform=transform, download=True)
    mnist_test_ds = MNIST(root="./data", train=True, transform=transform, download=True)

    mnist_train_ds_len = len(mnist_train_ds)
    mnist_test_ds_len = len(mnist_test_ds)

    train_ds = mnist_train_ds
    test_ds = mnist_test_ds

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
    )

    class FeatureExtracter(nn.Module):

        def __init__(self):
            super(FeatureExtracter, self).__init__()
            self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
            self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()

        def forward(self, x: Tensor):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = x.view(-1, 8192)
            return x

    class NewFE(nn.Module):
        def __init__(self, in_dim, out_dim) -> None:
            super(NewFE, self).__init__()
            self.fc1 = nn.Linear(in_dim, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, out_dim)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.sigmod = nn.Sigmoid()

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().train(False).to(output_device)
    newfe = NewFE(8192, 50).to(output_device).train(False)
    myinversion = Inversion(50).train(False).to(output_device)

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(
        torch.load(open(feature_pkl, "rb"), map_location=output_device)
    )

    assert os.path.exists(newfe_pkl)
    newfe.load_state_dict(torch.load(open(newfe_pkl, "rb"), map_location=output_device))

    assert os.path.exists(inv_pkl)
    myinversion.load_state_dict(
        torch.load(open(inv_pkl, "rb"), map_location=output_device)
    )

    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            feature8192 = feature_extractor.forward(im)
            feature = newfe.forward(feature8192)
            rim = myinversion.forward(feature)
            save_image2(im.detach(), f"{log_dir}/priv/input/{i}.png", nrow=4)
            save_image2(rim.detach(), f"{log_dir}/priv/output/{i}.png", nrow=4)

        for i, (im, label) in enumerate(tqdm(test_dl, desc=f"aux")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            feature8192 = feature_extractor.forward(im)
            feature = newfe.forward(feature8192)
            rim = myinversion.forward(feature)
            save_image2(im.detach(), f"{log_dir}/aux/input/{i}.png", nrow=4)
            save_image2(rim.detach(), f"{log_dir}/aux/output/{i}.png", nrow=4)

    writer.close()
