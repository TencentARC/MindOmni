import os
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import numpy as np


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

    arr = arr[crop_y1:arr.shape[0] - crop_y2, crop_x1:arr.shape[1] - crop_x2]
    return Image.fromarray(arr)


class OmniGenProcessor:
    def __init__(self, max_image_size: int = 1024):
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: crop_arr(pil_image, max_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           allow_patterns="*.json")
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(text_tokenizer)

    def process_image(self, image):
        image = Image.open(image).convert('RGB')
        return self.image_transform(image)

    def __call__(self,
                 context_hidden_state: List[torch.tensor],
                 neg_context_hidden_state: List[torch.tensor],
                 height: int = 1024,
                 width: int = 1024,
                 separate_cfg_input: bool = False,
                 ) -> Dict:

        input_data = []
        for i in range(len(context_hidden_state)):
            cur_context_hidden_state = context_hidden_state[i]
            cur_neg_context_hidden_state = neg_context_hidden_state[i]

            input_data.append((cur_context_hidden_state, cur_neg_context_hidden_state, [height, width]))

        if separate_cfg_input:
            return self.separate_collator(input_data)
        return self.collator(input_data)


class OmniGenCollator:
    def __init__(self, pad_token_id=2, llm_pad_token_id=151643, hidden_size=3072):
        self.llm_pad_token_id = llm_pad_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def create_position(self, attention_mask, num_tokens_for_output_images):
        position_ids = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            temp_position = [0] * (text_length - temp_l) + [i for i in range(temp_l + img_length + 1)]  # we add a time embedding into the sequence, so add one more token
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)

    def create_connector_position(self, llm_2d_attention_mask):
        position_ids = []
        text_length = llm_2d_attention_mask.size(-1)
        # img_length = max(num_tokens_for_output_images)
        for batch_idx, mask in enumerate(llm_2d_attention_mask):
            temp_l = torch.sum(llm_2d_attention_mask[batch_idx])
            # temp_position = [0]*(text_length-temp_l) + [i for i in range(temp_l+img_length+1)] # we add a time embedding into the sequence, so add one more token
            temp_position = [0] * (text_length - temp_l) + [i for i in range(temp_l)]  # only condition for mllm like qwen
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)

    def create_mask(self, attention_mask, num_tokens_for_output_images):
        extended_mask = []
        padding_images = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1  # we add a time embedding into the sequence, so add one more token
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = text_length - temp_l

            temp_mask = torch.tril(torch.ones(size=(temp_l + 1, temp_l + 1)))

            image_mask = torch.zeros(size=(temp_l + 1, img_length))
            temp_mask = torch.cat([temp_mask, image_mask], dim=-1)

            image_mask = torch.ones(size=(img_length, temp_l + img_length + 1))
            temp_mask = torch.cat([temp_mask, image_mask], dim=0)

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l + 1 + img_length, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            true_img_length = num_tokens_for_output_images[inx]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask[:, -pad_img_length:] = 0
                temp_padding_imgs = torch.zeros(size=(1, pad_img_length, self.hidden_size))
            else:
                temp_padding_imgs = None

            extended_mask.append(temp_mask.unsqueeze(0))
            padding_images.append(temp_padding_imgs)
            inx += 1
        return torch.cat(extended_mask, dim=0), padding_images

    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        for b_inx in image_sizes.keys():
            for start_inx, end_inx in image_sizes[b_inx]:
                attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1

        return attention_mask

    def pad_input(self, context_hidden_state):
        # pad_token_id = self.llm_pad_token_id  # 151642 <|endoftext|> in qwen2.5vl
        max_l = max([x.shape[1] for x in context_hidden_state])
        attention_mask = []

        for i in range(len(context_hidden_state)):
            temp_hidden = context_hidden_state[i]
            temp_l = temp_hidden.shape[1]
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1] * max_l)
            else:
                attention_mask.append([0] * pad_l + [1] * temp_l)

        return torch.LongTensor(attention_mask)

    def process_mllm_input(self, context_hidden_state, target_img_size):
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0] * img_size[1] // 16 // 16)

        llm_2d_attention_mask = self.pad_input(context_hidden_state)
        connector_position_ids = self.create_connector_position(llm_2d_attention_mask)
        llm_position_ids = self.create_position(llm_2d_attention_mask, num_tokens_for_output_images)
        llm_attention_mask, _ = self.create_mask(llm_2d_attention_mask, num_tokens_for_output_images)

        return llm_2d_attention_mask, connector_position_ids, llm_attention_mask, llm_position_ids


class OmniGenSeparateCollator(OmniGenCollator):
    def __call__(self, features):
        context_hidden_state = [f[0] for f in features]
        neg_context_hidden_state = [f[1] for f in features]
        target_img_size = [f[2] for f in features]

        all_context_hidden_state, all_connector_attention_mask, all_connector_position_ids, all_llm_attention_mask, all_llm_position_ids = [], [], [], [], []
        connector_attention_mask, connector_position_ids, llm_attention_mask, llm_position_ids = self.process_mllm_input(context_hidden_state, target_img_size)

        all_context_hidden_state.append(context_hidden_state[0])
        all_connector_attention_mask.append(connector_attention_mask)
        all_connector_position_ids.append(connector_position_ids)
        all_llm_attention_mask.append(llm_attention_mask)
        all_llm_position_ids.append(llm_position_ids)

        if neg_context_hidden_state[0] is not None:
            connector_attention_mask, connector_position_ids, llm_attention_mask, llm_position_ids = self.process_mllm_input(neg_context_hidden_state, target_img_size)
            all_context_hidden_state.append(neg_context_hidden_state[0])
            all_connector_attention_mask.append(connector_attention_mask)
            all_connector_position_ids.append(connector_position_ids)
            all_llm_attention_mask.append(llm_attention_mask)
            all_llm_position_ids.append(llm_position_ids)

        data = {
            "context_hidden_state": all_context_hidden_state,
            "connector_attention_mask": all_connector_attention_mask,
            "connector_position_ids": all_connector_position_ids,
            "llm_attention_mask": all_llm_attention_mask,
            "llm_position_ids": all_llm_position_ids,
        }
        return data
