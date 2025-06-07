#  This code is based on OmniGen
from typing import List, Union
import gc

from PIL import Image
import torch
try:
    import torch_npu
except Exception as e:
    print(e)
from diffusers.models import AutoencoderKL
from diffusers.utils import logging
import torch.nn as nn
from .processor import OmniGenProcessor
from .model import OmniGen
from .scheduler import OmniGenScheduler


logger = logging.get_logger(__name__)


class ImageDecoderPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        connector: nn.Module,
        processor: OmniGenProcessor,
        device: Union[str, torch.device] = None,
    ):
        self.vae = vae
        self.model = model
        self.connector = connector
        self.processor = processor
        self.device = device

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch_npu.npu.is_available():
                self.device = torch.device("npu")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                logger.info("Don't detect any available GPUs, using CPU instead, this may take long time to generate image!!!")
                self.device = torch.device("cpu")

        # self.model.to(torch.bfloat16)
        self.model.eval()
        self.vae.eval()

        self.model_cpu_offload = False

    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)
        self.device = device

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        x = x.to(dtype)
        return x

    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def enable_model_cpu_offload(self):
        self.model_cpu_offload = True
        self.model.to("cpu")
        self.vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear VRAM
        elif torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()  # Clear VRAM
        gc.collect()  # Run garbage collection to free system RAM

    def disable_model_cpu_offload(self):
        self.model_cpu_offload = False
        self.model.to(self.device)
        self.vae.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        context_hidden_state: Union[str, List[str]] = None,
        neg_context_hidden_state: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        output_type: str = "pil",
        tqdm_disable: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800).
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800).
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output.
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            output_type (`str`, *optional*, defaults to "pil"):
                The type of the output image, which can be "pt" or "pil"
        Examples:

        Returns:
            A list with the generated images.
        """
        # check inputs:
        assert height % 16 == 0 and width % 16 == 0, "The height and width must be a multiple of 16."
        if context_hidden_state is not None and not isinstance(context_hidden_state, list):
            context_hidden_state = [context_hidden_state]
            neg_context_hidden_state = [neg_context_hidden_state]

        # set model and processor
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(max_image_size=max_input_image_size)
        self.model.to(dtype)
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()

        input_data = self.processor(context_hidden_state, neg_context_hidden_state, height=height, width=width, separate_cfg_input=separate_cfg_infer)

        num_prompt = len(context_hidden_state)
        num_cfg = 1
        latent_size_h, latent_size_w = height // 8, width // 8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents] * (1 + num_cfg), 0).to(dtype)

        model_kwargs = dict(cfg_scale=guidance_scale,
                            use_kv_cache=use_kv_cache,
                            offload_model=offload_model,
                            )
        # obtain the qwen feature
        # if self.llm_processor is not None:
        llm_input_embeds = []
        with torch.no_grad():
            # for seperate cfg infer mode
            for i in range(len(input_data['context_hidden_state'])):

                context_hidden_state = input_data['context_hidden_state'][i]
                hidden_states = self.connector[0](context_hidden_state)
                cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)

                mask_func = self.model.llm._update_causal_mask
                cond_causal_mask = mask_func(
                    input_data['connector_attention_mask'][i].to(self.device), hidden_states, cache_position, None, None)
                for decoder_layer in self.connector[1:]:
                    layer_out = decoder_layer(
                        hidden_states,
                        attention_mask=cond_causal_mask,
                        position_ids=input_data['connector_position_ids'][i].to(self.device),
                    )
                    hidden_states = layer_out[0]

                llm_input_embeds.append(hidden_states)

            # import ipdb; ipdb.set_trace()
            model_kwargs['llm_input_embeds'] = llm_input_embeds
            model_kwargs['llm_attention_mask'] = self.move_to_device(input_data['llm_attention_mask'])
            model_kwargs['llm_position_ids'] = self.move_to_device(input_data['llm_position_ids'])

        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        if self.model_cpu_offload:
            for name, param in self.model.named_parameters():
                if 'layers' in name and 'layers.0' not in name:
                    param.data = param.data.cpu()
                else:
                    param.data = param.data.to(self.device)
            for buffer_name, buffer in self.model.named_buffers():
                setattr(self.model, buffer_name, buffer.to(self.device))
        # else:
        #     self.model.to(self.device)

        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache, tqdm_disable=tqdm_disable)
        samples = samples.chunk((1 + num_cfg), dim=0)[0]

        if self.model_cpu_offload:
            self.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear VRAM
            elif torch_npu.npu.is_available():
                torch_npu.npu.empty_cache()  # Clear VRAM
            gc.collect()

        self.vae.to(self.device)
        samples = samples.to(torch.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor
        samples = self.vae.decode(samples).sample

        if self.model_cpu_offload:
            self.vae.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear VRAM
            elif torch_npu.npu.is_available():
                torch_npu.npu.empty_cache()  # Clear VRAM
            gc.collect()

        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        if output_type == "pt":
            output_images = samples
        else:
            output_samples = (samples * 255).to("cpu", dtype=torch.uint8)
            output_samples = output_samples.permute(0, 2, 3, 1).numpy()
            output_images = []
            for i, sample in enumerate(output_samples):
                output_images.append(Image.fromarray(sample))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear VRAM
        elif torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()  # Clear VRAM
        gc.collect()              # Run garbage collection to free system RAM

        return output_images
