import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


def save_img(tensor):
    plt.figure()
    img_tensors = np.transpose(tensor, (0, 2, 3, 1))

    for i, img_tensor in enumerate(img_tensors):
        # min max scaling
        min = img_tensor.min()
        max = img_tensor.max()
        tensor_minmax = (img_tensor - min) / (max - min)

        tensor_minmax = (tensor_minmax * 255).astype(np.uint8)

        pil_img = Image.fromarray(tensor_minmax)
        pil_img.save(f'.\\output\\g_image_{i + 1}.png')


def generate():
    manual_seed = random.randint(1, 10000)
    print('Random Seed: ', manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    ngpu = 1
    num_images = 10

    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    print('device: ', device)

    netG = torch.load('.\\output\\model\\generator.pth')
    noise = torch.randn(num_images, 100, 1, 1, device=device)

    g_result = netG(noise)
    save_img(g_result.to('cpu').detach().numpy())

if __name__ == '__main__':
    generate()