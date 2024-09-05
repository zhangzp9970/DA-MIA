import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 50
    log_epoch = 4
    class_num = 291
    root_dir = "./logZZPMAIN"
    dataset_dir = "./datasets/facescrub common"
    h = 64
    w = 64

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"main FaceScrub colourful flip",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose(
        [
            Resize((h, w)),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )

    ds = ImageFolder(root=dataset_dir, transform=transform)

    ds_len = len(ds)

    train_ds, test_ds = random_split(ds, [ds_len * 6 // 7, ds_len - ds_len * 6 // 7])

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    # for train
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
    )
    # for evaluate
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
    )

    train_dl_len = len(train_dl)
    test_dl_len = len(test_dl)

    class FeatureExtracter(nn.Module):
        def __init__(self):
            super(FeatureExtracter, self).__init__()
            self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
            self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
            self.conv4 = nn.Conv2d(512, 1024, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(512)
            self.bn4 = nn.BatchNorm2d(1024)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.mp4 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()

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
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            x = self.relu4(x)
            x = x.view(-1, 16384)
            return x

    class CLS(nn.Module):
        def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
            super(CLS, self).__init__()
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)

        def forward(self, x):
            x = self.bottleneck(x)
            x = self.fc(x)
            return x

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super(Classifier, self).__init__()
            self.feature_extractor = FeatureExtracter()
            self.cls = CLS(16384, class_num, bottle_neck_dim=2650)

        def forward(self, x):
            x = self.feature_extractor(x)
            x = self.cls(x)
            return x

    myclassifier = Classifier().train(True).to(output_device)

    optimizer = optim.Adam(
        myclassifier.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, (im, label) in enumerate(tqdm(train_dl, desc=f"epoch {epoch_id}")):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            out = myclassifier.forward(im)
            ce = nn.CrossEntropyLoss()(out, label)
            loss = ce
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            train_ca = ClassificationAccuracy(class_num)
            after_softmax = F.softmax(out, dim=-1)
            predict = torch.argmax(after_softmax, dim=-1)
            train_ca.accumulate(label=label, predict=predict)
            acc_train = train_ca.get()
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("acc_training", acc_train, epoch_id)
            with open(
                os.path.join(log_dir, f"feature_extractor_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myclassifier.feature_extractor.state_dict(), f)
            with open(os.path.join(log_dir, f"cls_{epoch_id}.pkl"), "wb") as f:
                torch.save(myclassifier.cls.state_dict(), f)

            with torch.no_grad():
                myclassifier.eval()
                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(train_dl, desc="testing train")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    out = myclassifier.forward(im)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("train loss", celossavg, epoch_id)
                writer.add_scalar("acc_train", acc_test, epoch_id)

                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(test_dl, desc="testing test")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    out = myclassifier.forward(im)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("test loss", celossavg, epoch_id)
                writer.add_scalar("acc_test", acc_test, epoch_id)

    writer.close()
