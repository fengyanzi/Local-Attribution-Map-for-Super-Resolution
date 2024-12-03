import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch
from PIL import Image
from matplotlib import pyplot as plt
from scipy import stats


def PIL2Tensor(pil_image):
    if isinstance(pil_image, list):
        pils = []
        for img in pil_image:
            pils.append(F.to_tensor(img))
        return torch.stack(pils)
    return F.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return F.to_pil_image(tensor_image.detach(), mode=mode)



def cv2_to_pil(img):

    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target


def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)


def vis_saliency(map, zoomin=1):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('gnuplot2')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(map)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    Img = Img.resize((s1 * zoomin, s2 * zoomin), Image.NEAREST)
    return Img.convert('RGB')


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil


def grad_abs_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad.sum(axis=0))
    grad_2d = np.sqrt(grad_2d)
    #平方根


    grad_max = grad_2d.max()

    grad_norm = grad_2d / grad_max
    return grad_norm


def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
    return tensor.view((1,) + tensor.size())



def interpolation(x, x_prime, fold, mode='linear'):
    diff = x - x_prime
    l = np.linspace(0, 1, fold).reshape((fold, 1, 1, 1))
    interp_list = l * diff + x_prime
    return interp_list


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)
