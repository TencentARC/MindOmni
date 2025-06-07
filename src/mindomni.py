from .mllm import MindOmniMLLM
from .image_decoder import OmniGen
import torch.nn as nn
from .image_decoder import Phi3DecoderLayer, ImageDecoderPipeline, OmniGenProcessor
import os
import torch
from safetensors.torch import load_file
from typing import Union
from diffusers.utils import logging
from diffusers.models import AutoencoderKL
from transformers import AutoProcessor
import re
from qwen_vl_utils import process_vision_info
try:
    import torch_npu
except Exception as e:
    print(e)

logger = logging.get_logger(__name__)


class MindOmniConnector(nn.Module):
    def __init__(self, pre_config, post_config, layer_num: int = 2):
        super().__init__()
        connector_decoder = nn.ModuleList(
            [Phi3DecoderLayer(post_config, layer_idx) for layer_idx in range(layer_num)]
        )
        self.connector = nn.ModuleList(
            [nn.Linear(pre_config.hidden_size, post_config.hidden_size)]  # qwen2.5vl-7b: 3584
        )
        self.connector.extend(connector_decoder)


class MindOmni:
    def __init__(self, mllm, image_decoder, connector, vae, processor, mllm_processor, device: Union[str, torch.device] = None):
        self.mllm = mllm
        self.image_decoder = image_decoder
        self.connector = connector
        self.vae = vae
        self.processor = processor
        self.mllm_processor = mllm_processor

        self.vae.to(torch.float32)
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

    @classmethod
    def from_pretrained(cls, model_path):
        mllm = MindOmniMLLM.from_pretrained(os.path.join(model_path, 'mllm'))
        image_decoder = OmniGen.from_pretrained(os.path.join(model_path, 'image_decoder'))
        connector = MindOmniConnector(mllm.config, image_decoder.llm.config, 2).connector
        connector_state = load_file(os.path.join(model_path, 'connector.safetensors'))
        connector.load_state_dict(connector_state)
        vae = AutoencoderKL.from_pretrained(os.path.join(model_path, "vae"))
        processor = OmniGenProcessor.from_pretrained(os.path.join(model_path, 'image_decoder'))
        mllm_processor = AutoProcessor.from_pretrained(os.path.join(model_path, 'mllm'))
        logger.info("Preparing MindOmni")
        return cls(mllm, image_decoder, connector, vae, processor, mllm_processor)

    def to(self, device: Union[str, torch.device] = None, dtype: Union[str, torch.device] = None):
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.mllm.to(device)
            self.image_decoder.to(device)
            self.connector.to(device)
            self.vae.to(device)
            self.device = device
        if dtype is not None:
            self.mllm.to(dtype)
            self.image_decoder.to(dtype)
            self.connector.to(dtype)

    def eval(self):
        self.mllm.eval()
        self.image_decoder.eval()
        self.connector.eval()
        self.vae.eval()

    @torch.no_grad()
    def get_mllm_hidden_state(self, user_input, input_images, do_sample, temperature, max_new_tokens, only_understand=False, use_cot=False):
        input_llm_images = input_images
        processor = self.mllm_processor
        model = self.mllm
        if only_understand or not use_cot:
            system_prompt = (
                "You are a helpful assistant."
            )
        else:
            system_prompt = (
                "You are a helpful assistant. When the user requests an image, the assistant "
                "first thinks about the reasoning process in the mind and then provides the user with concise prompt as the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                "<think> reasoning process here </think><answer> answer here </answer>."
            )

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate an image according to the following instructions\n"},
                    {"type": "text", "text": user_input},
                ],
            }
        ]

        if input_llm_images is not None:
            if only_understand:
                assert len(input_llm_images) == 1, "only support single image when multimodal understanding"
                messages[1]['content'][0] = {"type": "image", "image": input_llm_images[0]}
            else:
                user_input = f'<img><|image_1|></img> {user_input}'
                messages[1]['content'][1] = {"type": "text", "text": user_input}
                image_tags = re.findall(r'<\|image_\d+\|>', messages[1]['content'][1]['text'])
                image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
                pattern = r"<img><\|image_\d+\|></img>"
                prompt_chunks = [chunk for chunk in re.split(pattern, messages[1]['content'][1]['text'])]
                assert len(prompt_chunks) == len(input_llm_images) + 1
                new_content = []
                for idx, per_prompt in enumerate(prompt_chunks):
                    if idx != len(prompt_chunks) - 1:
                        item_text = {"type": "text", "text": per_prompt}
                        # resized_height, resized_width = input_images_shape[image_ids[idx] - 1]
                        image_path = input_llm_images[image_ids[idx] - 1]
                        # item_vit = {"type": "image", "image": image_path, "resized_height": resized_height, "resized_width": resized_width}
                        item_vit = {"type": "image", "image": image_path}
                        item_tag = {"type": "text", "text": f"<img>{image_tags[idx]}</img>"}
                        new_content.append(item_text)
                        new_content.append(item_vit)
                        new_content.append(item_tag)
                    else:
                        item_text = {"type": "text", "text": per_prompt}
                        new_content.append(item_text)
                messages[1]['content'] = messages[1]['content'][:1] + new_content

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("npu")

        if use_cot:
            # Inference: Generation of the output
            temperature = temperature if do_sample else None
            generated_dict = model.generate(**inputs, do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, output_hidden_states=True, return_dict_in_generate=True)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_dict.sequences)
            ]
            output_hidden_state = [hidden_state[-1] for hidden_state in generated_dict.hidden_states]
            context_hidden_state = torch.cat(output_hidden_state, dim=1)

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            prompt_ = output_text[0]

            assistant_content = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": prompt_},
                    ],
                }
            ]

            messages += assistant_content
        else:
            prompt_ = user_input
            context_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]
        return messages, prompt_, context_hidden_state

    def generate_image(self, height, width, guidance_scale, inference_steps, separate_cfg_infer, offload_model, seed, max_input_image_size,
                       text, NEGATIVE_PROMPT, input_llm_images, do_sample, temperature, max_new_tokens, only_understand, use_cot=False):
        gen_pipe = ImageDecoderPipeline(self.vae, self.image_decoder, self.connector, self.processor)
        message, prompt_, context_hidden_state = self.get_mllm_hidden_state(text, input_llm_images, do_sample, temperature, max_new_tokens, only_understand, use_cot=use_cot)
        neg_message, neg_prompt_, neg_context_hidden_state = self.get_mllm_hidden_state(NEGATIVE_PROMPT, None, do_sample, temperature, max_new_tokens, only_understand, use_cot=False)
        print(message)
        output = gen_pipe(
            context_hidden_state=context_hidden_state,
            neg_context_hidden_state=neg_context_hidden_state,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            separate_cfg_infer=separate_cfg_infer,
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        return output, prompt_

    def generate_text(self, text, input_llm_images, do_sample, temperature, max_new_tokens, only_understand):
        _, answer, _ = self.get_mllm_hidden_state(text, input_llm_images, do_sample, temperature, max_new_tokens, only_understand=True, use_cot=True)
        return answer
