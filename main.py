import argparse
import cv2
import torch
from lam.utils import Tensor2PIL, PIL2Tensor, cv2_to_pil, pil_to_cv2, vis_saliency, vis_saliency_kde, grad_abs_norm, \
    prepare_images
from lam.core import attribution_objective, Path_gradient, attr_grad, GaussianBlurPath
from lam.core import saliency_map_PG as saliency_map
# 测试不同模型请更改此处
# Please change here to test different models
from model.SRmodel.edsr import EDSR

def main(modelpath, imgpath, w, h, window_size, fold, sigma, l, alpha, zoomfactor, kde, output_dir):
    # 测试不同模型请更改此处
    # Please change here to test different models
    model = EDSR().to("cuda").eval()

    # 加载模型
    # Load the model
    model.load_state_dict(torch.load(modelpath))

    # 准备图像
    # Prepare images
    img_lr, img_hr = prepare_images(imgpath, scale=zoomfactor)
    tensor_lr = PIL2Tensor(img_lr)[:3]
    # tensor_hr = PIL2Tensor(img_hr)[:3]

    # 在高分辨率图像上绘制矩形
    # Draw rectangle on high resolution image
    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    # 归因目标函数和高斯模糊路径函数, 详情请阅读IG与LAM这两篇论文
    # Attribution objective and Gaussian blur path function
    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)

    # 计算梯度和显著性图
    # Compute gradients and saliency maps
    interpolated_grad_numpy, result_numpy, _ = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                             gaus_blur_path_func)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)

    # 可视化
    # Visualize saliency
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=zoomfactor)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)

    # 使用KDE可视化时生成额外的显著性图
    # Generate additional saliency map when using KDE for visualization
    if kde:
        saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy, zoomin=zoomfactor)
        blend_kde_and_input = cv2_to_pil(
            pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
        combined_images = [position_pil, saliency_image_abs, blend_abs_and_input, blend_kde_and_input,
                           Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
        image_names = ['window_position.png', 'saliency_abs.png', 'blend_abs.png', 'blend_kde.png', 'result.png']
    else:
        combined_images = [position_pil, saliency_image_abs, blend_abs_and_input,
                           Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
        image_names = ['window_position.png', 'saliency_abs.png', 'blend_abs.png', 'result.png']

    # 使用列表中的名称保存图片
    # Save images with names from the list
    for i, image in enumerate(combined_images):
        image.save(f"{output_dir}/{image_names[i]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save saliency maps.")
    parser.add_argument('--modelpath', type=str, default='model/weights/EDSR-64-16.pth',
                        help='Path to the model weights. 模型权重路径。')
    parser.add_argument('--imgpath', type=str, default='testimg/test.png',
                        help='Path to the input image. 测试图像路径。')
    parser.add_argument('--w', type=int, default=50, help='The x coordinate of the select patch. 选择区域的x坐标。')
    parser.add_argument('--h', type=int, default=300, help='The y coordinate of the select patch. 选择区域的y坐标。')
    parser.add_argument('--window_size', type=int, default=32, help='Size of the window. 区域大小。')
    parser.add_argument('--fold', type=int, default=50, help='Number of folds. 路径积分步数，越高越接近真实，默认50，不了解不建议修改。')
    parser.add_argument('--sigma', type=float, default=1.2, help='Sigma value for Gaussian blur. 路径积分参数，不了解不建议修改。')
    parser.add_argument('--l', type=int, default=9, help='Parameter l for GaussianBlurPath. 路径积分参数，不了解不建议修改。')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha value for blending. 混合时的alpha值。')
    parser.add_argument('--zoomfactor', type=int, default=4, help='Zoom factor for images. 图像缩放因子,SR倍率。')
    parser.add_argument('--kde', default=False, help='Whether to use KDE for visualization. 是否使用KDE进行可视化。对电脑性能考验较大，时间较长！')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the output images. 输出图片目录。')

    args = parser.parse_args()

    main(args.modelpath, args.imgpath, args.w, args.h, args.window_size, args.fold, args.sigma, args.l, args.alpha,
         args.zoomfactor, args.kde, args.output_dir)