import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import (
    Init,
    MMD,
    save_image2,
)
from piq import SSIMLoss

if __name__ == "__main__":
    batch_size = 128
    train_epoches = 100
    log_epoch = 4
    class_num = 291
    root_dir = "./logZZPMAIN.attackcolorful"
    feature_pkl = "/path/to/target/model/trained/using/FaceScrub/feature_extractor.pkl"
    cls_pkl = "/path/to/target/model/trained/using/FaceScrub/cls.pkl"
    priv_dataset_dir = "./datasets/facescrub common"
    aux_dataset_dir = "./datasets/img_align_celeba_common_crop"
    h = 64
    w = 64
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"FaceScrub DATTACK C2F colorful newfe 8192 flip",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir, model_dir = init.get_log_dir()
    data_workers = 0

    ############################easydl##################################

    class GradientReverseLayer(torch.autograd.Function):
        """
        usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

            x = Variable(torch.ones(1, 2), requires_grad=True)
            grl = GradientReverseLayer.apply
            y = grl(0.5, x)

            y.backward(torch.ones_like(y))

            print(x.grad)

        """

        @staticmethod
        def forward(ctx, coeff, input):
            ctx.coeff = coeff
            # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
            return input.view_as(input)

        @staticmethod
        def backward(ctx, grad_outputs):
            coeff = ctx.coeff
            return None, -coeff * grad_outputs

    class GradientReverseModule(nn.Module):
        """
        wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

        usage::

            grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

            x = Variable(torch.ones(1), requires_grad=True)
            ans = []
            for _ in range(10000):
                x.grad = None
                y = grl(x)
                y.backward()
                ans.append(variable_to_numpy(x.grad))

            plt.plot(list(range(10000)), ans)
            plt.show() # you can see gradient change from 0 to -1
        """

        def __init__(self, scheduler):
            super(GradientReverseModule, self).__init__()
            self.scheduler = scheduler
            self.register_buffer("global_step", torch.zeros(1))
            self.coeff = 0.0
            self.grl = GradientReverseLayer.apply

        def forward(self, x):
            self.coeff = self.scheduler(self.global_step.item())
            if self.training:
                self.global_step += 1.0
            return self.grl(self.coeff, x)

    def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
        """
        change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
        A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

        =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

            from matplotlib import pyplot as plt

            ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
            xs = [x for x in range(10000)]

            plt.plot(xs, ys)
            plt.show()

        """
        ans = A + (2.0 / (1 + np.exp(-gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
        return float(ans)

    def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
        """
        change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
        as known as inv learning rate sheduler in caffe,
        see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

        the default gamma and power come from <Domain-Adversarial Training of Neural Networks>

        code to see how it changes(decays to %20 at %10 * max_iter under default arg)::

            from matplotlib import pyplot as plt

            ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
            xs = [x for x in range(10000)]

            plt.plot(xs, ys)
            plt.show()

        """
        return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (-power))

    class OptimWithSheduler:
        """
        usage::

            op = optim.SGD(lr=1e-3, params=net.parameters()) # create an optimizer
            scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=100, power=1, max_iter=100) # create a function
            that receives two keyword arguments:step, initial_lr
            opw = OptimWithSheduler(op, scheduler) # create a wrapped optimizer
            with OptimizerManager(opw): # use it as an ordinary optimizer
                loss.backward()
        """

        def __init__(self, optimizer, scheduler_func):
            self.optimizer = optimizer
            self.scheduler_func = scheduler_func
            self.global_step = 0.0
            for g in self.optimizer.param_groups:
                g["initial_lr"] = g["lr"]

        def zero_grad(self):
            self.optimizer.zero_grad()

        def step(self):
            for g in self.optimizer.param_groups:
                g["lr"] = self.scheduler_func(
                    step=self.global_step, initial_lr=g["initial_lr"]
                )
            self.optimizer.step()
            self.global_step += 1

    ####################################################################

    transform = Compose([Resize((h, w)), RandomHorizontalFlip(), ToTensor()])

    priv_ds = ImageFolder(root=priv_dataset_dir, transform=transform)
    aux_ds = ImageFolder(root=aux_dataset_dir, transform=transform)

    priv_ds_len = len(priv_ds)
    aux_ds_len = len(aux_ds)

    priv_train_ds, priv_test_ds = random_split(
        priv_ds, [priv_ds_len * 6 // 7, priv_ds_len - priv_ds_len * 6 // 7]
    )

    aux_train_ds, aux_test_ds = random_split(
        aux_ds, [aux_ds_len * 6 // 7, aux_ds_len - aux_ds_len * 6 // 7]
    )

    source_train_ds = aux_train_ds
    source_test_ds = aux_test_ds

    target_train_ds = priv_train_ds
    target_test_ds = priv_test_ds

    source_train_ds_len = len(source_train_ds)
    source_test_ds_len = len(source_test_ds)

    target_train_ds_len = len(target_train_ds)
    target_test_ds_len = len(target_test_ds)

    print(source_train_ds_len)
    print(source_test_ds_len)

    print(target_train_ds_len)
    print(target_test_ds_len)

    source_train_dl = DataLoader(
        dataset=source_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
    )
    source_test_dl = DataLoader(
        dataset=source_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
    )
    target_train_dl = DataLoader(
        dataset=target_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
    )
    target_test_dl = DataLoader(
        dataset=target_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
    )

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

    class NewFE(nn.Module):
        def __init__(self, in_dim, out_dim) -> None:
            super(NewFE, self).__init__()
            self.fc1 = nn.Linear(in_dim, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, out_dim)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    class AdversarialNetwork(nn.Module):
        def __init__(self, in_feature):
            super(AdversarialNetwork, self).__init__()
            self.grl = GradientReverseModule(
                lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000)
            )
            self.fc1 = nn.Linear(in_feature, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, 1)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.dropout1 = nn.Dropout()
            self.dropout2 = nn.Dropout()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.grl(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 1024, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(1024)
            self.bn2 = nn.BatchNorm2d(512)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
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
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.deconv5(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().to(output_device).train(False)
    newfe = NewFE(16384, 2650).to(output_device).train(True)
    myinversion = Inversion(2650).train(True).to(output_device)
    discriminator = AdversarialNetwork(2650).to(output_device).train(True)

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(
        torch.load(open(feature_pkl, "rb"), map_location=output_device)
    )

    feature_extractor.requires_grad_(False)

    def scheduler(step, initial_lr):
        return inverseDecaySheduler(
            step, initial_lr, gamma=10, power=0.75, max_iter=10000
        )

    optimizer = optim.Adam(
        myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    optimizer_newfe = OptimWithSheduler(
        optim.SGD(
            newfe.parameters(),
            lr=lr / 10,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        ),
        scheduler,
    )

    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(
            discriminator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        ),
        scheduler,
    )

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(
            tqdm(
                zip(source_train_dl, target_train_dl),
                desc=f"epoch {epoch_id}",
                total=min(len(source_train_dl), len(target_train_dl)),
            )
        ):
            im_source = im_source.to(output_device)
            label_source = label_source.to(output_device)
            im_target = im_target.to(output_device)
            label_target = label_target.to(output_device)
            bs, c, h, w = im_source.shape
            optimizer.zero_grad()
            optimizer_newfe.zero_grad()
            optimizer_discriminator.zero_grad()
            feature8192_source = feature_extractor.forward(im_source)
            feature8192_target = feature_extractor.forward(im_target)
            feature_source = newfe.forward(feature8192_source)
            feature_target = newfe.forward(feature8192_target)
            rim_source = myinversion.forward(feature_source)
            domain_source = discriminator.forward(feature_source)
            domain_target = discriminator.forward(feature_target)
            adv_loss = nn.BCELoss()(
                domain_source, torch.ones_like(domain_source)
            ) + nn.BCELoss()(domain_target, torch.zeros_like(domain_target))
            ssim = SSIMLoss()(rim_source, im_source)
            loss = ssim + adv_loss
            loss.backward()
            optimizer_newfe.step()
            optimizer_discriminator.step()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            writer.add_scalar("adv_loss", adv_loss, epoch_id)
            writer.add_scalar("ssim", ssim, epoch_id)
            writer.add_scalar("loss", loss, epoch_id)
            save_image2(im_source.detach(), f"{log_dir}/input/{epoch_id}.png")
            save_image2(rim_source.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(os.path.join(model_dir, f"newfe_{epoch_id}.pkl"), "wb") as f:
                torch.save(newfe.state_dict(), f)
            with open(
                os.path.join(model_dir, f"discriminator_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(discriminator.state_dict(), f)
            with open(
                os.path.join(model_dir, f"myinversion_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myinversion.state_dict(), f)

            with torch.no_grad():
                newfe.eval()
                myinversion.eval()
                discriminator.eval()
                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(
                    tqdm(source_train_dl, desc="aux train")
                ):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature = newfe.forward(feature8192)
                    rim = myinversion.forward(feature)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("aux ssim", ssimlossavg, epoch_id)

                r = 0
                ssimloss = 0
                for i, (im, label) in enumerate(
                    tqdm(target_train_dl, desc="priv train")
                ):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    feature8192 = feature_extractor.forward(im)
                    feature = newfe.forward(feature8192)
                    rim = myinversion.forward(feature)
                    ssim = SSIMLoss()(rim, im)
                    ssimloss += ssim

                ssimlossavg = ssimloss / r
                writer.add_scalar("priv ssim", ssimlossavg, epoch_id)

                r = 0
                mmdloss = 0
                for i, (
                    (im_source, label_source),
                    (im_target, label_target),
                ) in enumerate(
                    tqdm(
                        zip(source_train_dl, target_train_dl),
                        desc=f"MMD",
                        total=min(len(source_train_dl), len(target_train_dl)),
                    )
                ):
                    r += 1
                    im_source = im_source.to(output_device)
                    label_source = label_source.to(output_device)
                    im_target = im_target.to(output_device)
                    label_target = label_target.to(output_device)
                    bs, c, h, w = im_source.shape
                    feature8192_source = feature_extractor.forward(im_source)
                    feature8192_target = feature_extractor.forward(im_target)
                    feature_source = newfe.forward(feature8192_source)
                    feature_target = newfe.forward(feature8192_target)
                    mmd = MMD(feature_source, feature_target)
                    mmdloss += mmd

                mmdlossavg = mmdloss / r
                writer.add_scalar("mmd loss", mmdlossavg, epoch_id)

    writer.close()
