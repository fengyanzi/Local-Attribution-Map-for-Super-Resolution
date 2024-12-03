import numpy as np
import torch
import cv2
from lam.utils import _add_batch_one
from lam.utils import interpolation, isotropic_gaussian_kernel
from tqdm import tqdm


def attr_grad(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]
    return torch.sum(crop)


def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)

    return calculate_objective


def GaussianBlurPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        h, w, c = cv_numpy_image.shape
        kernel_interpolation = np.zeros((fold + 1, l, l))
        image_interpolation = np.zeros((fold, h, w, c))
        lambda_derivative_interpolation = np.zeros((fold, h, w, c))
        sigma_interpolation = np.linspace(sigma, 0, fold + 1)
        for i in tqdm(range(fold + 1)):
            kernel_interpolation[i] = isotropic_gaussian_kernel(l, sigma_interpolation[i])
        for i in tqdm(range(fold)):
            image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
            lambda_derivative_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, (
                    kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
            np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)

    return path_interpolation_func

# 另一种积分路径函数
def GaussianLinearPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        kernel = isotropic_gaussian_kernel(l, sigma)
        baseline_image = cv2.filter2D(cv_numpy_image, -1, kernel)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold,
                                                    axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
            np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)

    return path_interpolation_func


def Path_gradient(numpy_image, model, attr_objective, path_interpolation_func, cuda=True):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    if cuda:
        model = model.cuda()
    cv_numpy_image = np.moveaxis(numpy_image, 0, 2)
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in tqdm(range(image_interpolation.shape[0])):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)
        if cuda:
            result = model(_add_batch_one(img_tensor).cuda())
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        else:
            result = model(_add_batch_one(img_tensor))
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.cpu().detach().numpy())
    results_numpy = np.asarray(result_list)
    return grad_accumulate_list, results_numpy, image_interpolation


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]

