import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot
    """
    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        tx = -4
        ty = (img_half_width * 2 + 2) * y + (img_half_width + 2)
        plt.text(tx, ty, cls, ha="right")
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)
