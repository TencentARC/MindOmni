import logging

from PIL import Image
import torch
import numpy as np

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)




def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def crop_arr(pil_image, max_image_size):
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]    
    return Image.fromarray(arr)


def crop_by_max_pixels(pil_image, max_pixels=262144, max_side_length=1024, vae_scale_factor=16, resize_mode="default"):
    r"""
    Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
            should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
            tensor, it should have shape `[batch, channels, height, width]`.
        height (`Optional[int]`, *optional*, defaults to `None`):
            The height of the preprocessed image. If `None`, the height of the `image` input will be used.
        width (`Optional[int]`, *optional*, defaults to `None`):
            The width of the preprocessed image. If `None`, the width of the `image` input will be used.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the height and width, both resized to the nearest integer multiple of
            `vae_scale_factor`.
    """

    height = pil_image.height

    width = pil_image.width

    ratio = 1.0
    if max_side_length is not None:
        if height > width:
            max_side_length_ratio = max_side_length / height
        else:
            max_side_length_ratio = max_side_length / width

    cur_pixels = height * width
    max_pixels_ratio = (max_pixels / cur_pixels) ** 0.5
    ratio = min(max_pixels_ratio, max_side_length_ratio, 1.0)  # do not upscale input image

    new_height, new_width = int(height * ratio) // vae_scale_factor * vae_scale_factor, int(width * ratio) // vae_scale_factor * vae_scale_factor

    assert isinstance(pil_image, Image.Image), f"need preproccess image is PIL.Image but get {pil_image.dtype}"
    if resize_mode == "default":
        image = pil_image.resize(
            (new_width, new_height),
            resample=Image.LANCZOS,
            reducing_gap=None,
        )
    else:
        raise ValueError(f"resize_mode {resize_mode} is not supported")

    return image


def vae_encode(vae, x, weight_dtype):
    if x is not None:
        if vae.config.shift_factor is not None:
            x = vae.encode(x).latent_dist.sample()
            x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
        x = x.to(weight_dtype)
    return x

def vae_encode_list(vae, x, weight_dtype):
    latents = []
    for img in x:
        img = vae_encode(vae, img, weight_dtype)
        latents.append(img)
    return latents

