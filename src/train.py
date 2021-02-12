from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from model.generator import Generator
from model.discriminator import Discriminator

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_g_images(i, img_list):
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(f'.\\output\\image\\output_{str(i)}.png')

def train():
    # Random Seed
    manual_seed = random.randint(1, 10000)
    print('Random Seed: ', manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Parameter
    dataroot = 'E:\\datasets\\ukiyoe-1024'
    workers = 2
    batch_size = 128
    image_size = 64
    nz = 100
    num_epochs = 100

    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    # create dataset and dataloader
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # define device
    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    print('device: ', device)

    netG = Generator(ngpu).to(device)
    netG.apply(weight_init)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weight_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # training loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print('Starting Training Loop.')
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update D network
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)
            errG = criterion(output, label)

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch + 1, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        save_g_images(epoch, img_list)

    model_path_G = '.\\output\\model\\generator.pth'
    model_path_D = '.\\output\\model\\discriminator.pth'
    torch.save(netG, model_path_G)
    torch.save(netD, model_path_D)

    print('Finish training.')

if __name__ == '__main__':
    train()