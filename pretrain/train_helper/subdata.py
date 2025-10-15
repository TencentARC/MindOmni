import os
import datasets
# from datasets import load_dataset, ClassLabel, concatenate_datasets
from .utils import load_annotation
import torch
import numpy as np
import random
from PIL import Image
import json
import copy
# import torchvision.transforms as T
from torchvision import transforms
import pickle 
import re

from OmniGen import OmniGenProcessor
from OmniGen.processor import OmniGenCollator
from .webdataset_ import X2IWebDataset
from .webdataset_laion import ShortLongWebDataset


class SubDatasetFromJson(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file: str, 
        image_path: str,
        processer: OmniGenProcessor,
        image_transform,
        llm_processor=None,
        max_input_length_limit: int = 18000,
        condition_dropout_prob: float = 0.1,
        keep_raw_resolution: bool = True,
        llm_type: str = 'qwen2.5vl',
        epoch_length: int = None,
    ):
        
        self.image_transform = image_transform
        self.processer = processer
        self.condition_dropout_prob = condition_dropout_prob
        self.max_input_length_limit = max_input_length_limit
        self.keep_raw_resolution = keep_raw_resolution

        # self.data = load_dataset('json', data_files=json_file)['train']
        if image_path is not None:
            is_replace_image_file = True
        else:
            is_replace_image_file = False
        self.data = load_annotation(data_path=json_file, image_dir=image_path, is_replace_image_file=is_replace_image_file, key_list=['input_images', 'output_image'])
        self.image_path = image_path
        self.llm_type = llm_type
        self.llm_processor = llm_processor
        self.epoch_length = epoch_length

    def process_image(self, image_file):
        # if self.image_path is not None:
            # image_file = os.path.join(self.image_path, image_file)
        image = Image.open(image_file).convert('RGB')
        return self.image_transform(image)

    def get_example(self, index):
        example = self.data[index]
        
        instruction, input_images, output_image = example['instruction'], example['input_images'], example['output_image']
        qwen_instruction = example.get('instruction_qwen', None)
        think_content = example.get('think_content', None)
        if qwen_instruction is not None:
            short_instruction = instruction
            instruction = qwen_instruction
        else:
            short_instruction = None

        if random.random() < self.condition_dropout_prob:
            instruction = '<cfg>'
            input_images = None
        if input_images is not None:
            input_images_path = input_images
            input_images = [self.process_image(x) for x in input_images]

            # input_llm_images = [os.path.join(self.image_path, x) for x in input_images_path]
            input_llm_images = input_images_path
        else:
            input_llm_images = None

        mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images, llm_type=self.llm_type, llm_processor=self.llm_processor, short_instruction=short_instruction, input_llm_images=input_llm_images, think_content=think_content)

        output_image = self.process_image(output_image)
            
        if len(mllm_input['llm_input_ids']) > self.max_input_length_limit:
            print(example)
            raise RuntimeError(f"cur number of llm tokens={len(mllm_input['llm_input_ids'])}, larger than max_input_length_limit={self.max_input_length_limit}")

        return (mllm_input, output_image)

    def _attempt_get_data_info(self, index):
        try:
            return self.get_example(index)
        except Exception as e:  # noqa
            print(e)
            index = random.randint(0, len(self.data) - 1)
            return self._attempt_get_data_info(index)

        # try :
        #     return self.get_example(index)
        # except:
        #     print(index)

    def get_data_info(self, index):
        return self._attempt_get_data_info(index)
        
        # return self.t2i_webdataset.get_data_info(index)
        for _ in range(8):
            try:
                mllm_input, output_image = self.get_example(index)
                if len(mllm_input['input_ids']) > self.max_input_length_limit:
                    raise RuntimeError(f"cur number of tokens={len(mllm_input['input_ids'])}, larger than max_input_length_limit={self.max_input_length_limit}")
                return mllm_input, output_image
            except Exception as e:
                print("error when loading data: ", e)
                print(self.data[index])
                index = random.randint(0, len(self.data)-1)
        raise RuntimeError("Too many bad data.")
    

    def __len__(self):
        if self.epoch_length is not None:
            return self.epoch_length
        else:
            return len(self.data)



class TrainDataCollator(OmniGenCollator):
    def __init__(self, pad_token_id: int, llm_pad_token_id: int, hidden_size: int, keep_raw_resolution: bool):
        self.llm_pad_token_id = llm_pad_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.keep_raw_resolution = keep_raw_resolution

    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]

        output_images = [f[1].unsqueeze(0) for f in features]
        target_img_size = [[x.size(-2), x.size(-1)] for x in output_images]

        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes,\
            llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids = self.process_mllm_input(mllm_inputs, target_img_size)

        if not self.keep_raw_resolution:
            output_images = torch.cat(output_images, dim=0)
            if len(all_pixel_values) > 0:
                all_pixel_values = torch.cat(all_pixel_values, dim=0)
            else:
                all_pixel_values = None

        data = {"input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "input_pixel_values": all_pixel_values,
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        "output_images": output_images,
        "llm_padded_input_ids": llm_padded_input_ids,
        "llm_2d_attention_mask": llm_2d_attention_mask,
        "llm_attention_mask": llm_attention_mask,
        "llm_position_ids": llm_position_ids,
        "llm_inputs": llm_inputs,
        "llm_image_sizes": llm_image_sizes,
        "llm_vae_attention_mask": llm_vae_attention_mask,
        "llm_vae_position_ids": llm_vae_position_ids,
        }
        return data





