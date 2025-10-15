import os
import re
from typing import Dict, List
import json

import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info

from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
)
import copy as cp



class OmniGenProcessor:
    def __init__(self, 
                text_tokenizer, 
                max_image_size: int=1024):
        self.text_tokenizer = text_tokenizer
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
    
    def process_multi_modal_prompt(self, text, input_images, llm_type=None, llm_processor=None, input_llm_images=None, short_instruction=None, pure_text=None, think_content=None):
        '''
        input_images: PIL.Image for vae
        input_llm_images: image path for vit
        '''
        if llm_type == 'qwen2.5vl' and llm_processor is not None:
            # TODO: only support t2i mode and qwen2.5vl model
            if input_images is not None:
                input_images_shape = [[x.shape[1], x.shape[2]] for x in input_images]    # [h, w]
            else:
                input_images_shape = None
            llm_input_ids, llm_inputs, llm_img_inx, llm_text, new_text = self.add_prefix_instruction_llm(text, llm_processor, input_llm_images, input_images, input_images_shape, short_instruction=short_instruction, pure_text=pure_text, think_content=think_content)
            if short_instruction is not None:
                text = new_text     # if no image input: answer content for teacher model, else: short instruction 为了获得noise和mask等输入数据构建
            if pure_text is not None:   # only for inference mode, code below is invalid but neccessay
                text = llm_text
        else:
            llm_input_ids, llm_inputs, llm_img_inx = None, None, None
        text = self.add_prefix_instruction(text)
        # if pure_text:
        #     text = llm_text
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None, "llm_input_ids": llm_input_ids, "llm_inputs": llm_inputs}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)] 

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        
        input_images = [input_images[x-1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) -1:
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) *  input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx+size])
                all_input_ids.extend([0]*size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx, "llm_input_ids": llm_input_ids, "llm_inputs": llm_inputs, "llm_img_inx": llm_img_inx}


    def add_prefix_instruction(self, prompt):
        user_prompt = '<|user|>\n'
        generation_prompt = 'Generate an image according to the following instructions\n'
        assistant_prompt = '<|assistant|>\n<|diffusion|>'
        prompt_suffix = "<|end|>\n"
        prompt = f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"
        return prompt

    def add_prefix_instruction_llm(self, prompt, llm_processor, input_llm_images=None, input_images=None, input_images_shape=None, short_instruction=None, pure_text=None, think_content=None):
        generation_prompt = 'Generate an image according to the following instructions\n'
        new_omnigen_text = prompt
        if short_instruction is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": generation_prompt},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            system_prompt = (
                "You are a helpful assistant. When the user requests an image, the assistant "
                "first thinks about the reasoning process in the mind and then provides the user with concise prompt as the answer. "
                "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                "<think> reasoning process here </think><answer> answer here </answer>."
            )

            if think_content is not None:
                prompt_organize = f"<think> {think_content} </think><answer> {prompt}"
            else:
                prompt_organize = f"<think> reasoning process here </think><answer> {prompt}"

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
                        {"type": "text", "text": generation_prompt},
                        {"type": "text", "text": short_instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": prompt_organize},
                    ],
                }
            ]

        if pure_text is None:   # pure_text is a already messages that used when gradio app or grpo training
            if input_llm_images is not None and len(input_llm_images) > 0:
                if short_instruction is None:
                    image_tags = re.findall(r'<\|image_\d+\|>', messages[0]['content'][1]['text'])
                    image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
                    pattern = r"<img><\|image_\d+\|></img>"
                    prompt_chunks = [chunk for chunk in re.split(pattern, messages[0]['content'][1]['text'])]
                    assert len(prompt_chunks) == len(input_llm_images) + 1
                    new_content = []
                    for idx, per_prompt in enumerate(prompt_chunks):
                        if idx != len(prompt_chunks) - 1:
                            item_text = {"type": "text", "text": per_prompt}
                            resized_height, resized_width = input_images_shape[image_ids[idx] - 1]
                            image_path = input_llm_images[image_ids[idx] - 1]
                            item_vit = {"type": "image", "image": image_path, "resized_height": resized_height, "resized_width": resized_width}
                            item_tag = {"type": "text", "text": f"<img>{image_tags[idx]}</img>"}
                            new_content.append(item_text)
                            new_content.append(item_vit)
                            new_content.append(item_tag)
                        else:
                            item_text = {"type": "text", "text": per_prompt}
                            new_content.append(item_text)
                    messages[0]['content'] = messages[0]['content'][:1] + new_content
                else:
                    new_omnigen_text = short_instruction
                    image_tags = re.findall(r'<\|image_\d+\|>', messages[1]['content'][1]['text'])
                    image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
                    pattern = r"<img><\|image_\d+\|></img>"
                    prompt_chunks = [chunk for chunk in re.split(pattern, messages[1]['content'][1]['text'])]
                    assert len(prompt_chunks) == len(input_llm_images) + 1
                    new_content = []
                    for idx, per_prompt in enumerate(prompt_chunks):
                        if idx != len(prompt_chunks) - 1:
                            item_text = {"type": "text", "text": per_prompt}
                            resized_height, resized_width = input_images_shape[image_ids[idx] - 1]
                            image_path = input_llm_images[image_ids[idx] - 1]
                            item_vit = {"type": "image", "image": image_path, "resized_height": resized_height, "resized_width": resized_width}
                            item_tag = {"type": "text", "text": f"<img>{image_tags[idx]}</img>"}
                            new_content.append(item_text)
                            new_content.append(item_vit)
                            new_content.append(item_tag)
                        else:
                            item_text = {"type": "text", "text": per_prompt}
                            new_content.append(item_text)
                    messages[1]['content'] = messages[1]['content'][:1] + new_content

            text = llm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if short_instruction is not None:   # when using qwen tokenizer，train 'long short caption' mode, remove the last special tokens
                postfix = '<|im_end|>\n<|im_start|>assistant\n'
                text = text[:-len(postfix)] + '\n'
                # text = text[:-len(postfix)] + '<|im_end|>\n'    # When only training mllm, ensure that there is <im_end> at the end to train
        else:
            if len(pure_text) > 1:
                text = llm_processor.apply_chat_template(
                    pure_text, tokenize=False, add_generation_prompt=False
                )
                messages = pure_text
                text = text[:-len('<|im_end|>\n')] + '\n'  # When training grpo, only the text before </answer><|im_end|>\n is used as the condition, '</answer> is not included in the message and has been removed in advance'
            else:   # case in training online generate distillation, grab the answer content as instruction
                text = llm_processor.apply_chat_template(
                    pure_text, tokenize=False, add_generation_prompt=True
                )
                messages = pure_text

        image_inputs, video_inputs = process_vision_info(messages)

        llm_inputs = llm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            # return_tensors="pt",
        )

        if input_images is not None and len(input_images) != 0:
            pattern = r"<\|image_\d+\|>"
            assert video_inputs is None, "Only support Image"
            # prompt_chunks = [llm_processor(text=[chunk], images=[image_inputs[idx]], videos=None, padding=True).input_ids[0] for idx, chunk in enumerate(re.split(pattern, text))]
            prompt_chunks = []
            for idx, chunk in enumerate(re.split(pattern, text)):
                if idx != len(re.split(pattern, text)) - 1:
                    if image_inputs is None:    # for img_cfg with vae but without vit
                        temp = llm_processor(text=[chunk], images=None, videos=None, padding=True).input_ids[0]
                    else:
                        temp = llm_processor(text=[chunk], images=[image_inputs[idx]], videos=None, padding=True).input_ids[0]
                else:
                    temp = llm_processor(text=[chunk], images=None, videos=None, padding=True).input_ids[0]
                prompt_chunks.append(temp)

            image_tags = re.findall(pattern, text) 
            image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

            unique_image_ids = sorted(list(set(image_ids)))
            assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
            # total images must be the same as the number of image tags
            assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"

            input_images = [input_images[x-1] for x in image_ids]

            all_input_ids = []
            img_inx = []
            for i in range(len(prompt_chunks)):
                all_input_ids.extend(prompt_chunks[i])
                if i != len(prompt_chunks) -1:
                    start_inx = len(all_input_ids)
                    size = input_images[i].size(-2) * input_images[i].size(-1) // 16 // 16
                    img_inx.append([start_inx, start_inx + size])
                    all_input_ids.extend([0] * size)
            llm_inputs['input_ids'][0] = all_input_ids
        else:
            img_inx = None

        if 'pixel_values' not in llm_inputs.keys():
            return llm_inputs['input_ids'][0], None, img_inx, text, new_omnigen_text
        else:
            return llm_inputs['input_ids'][0], llm_inputs, img_inx, text, new_omnigen_text

    def __call__(self, 
                instructions: List[str], 
                input_images: List[List[str]] = None,
                height: int = 1024,
                width: int = 1024,
                negative_prompt: str = "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.",
                use_img_cfg: bool = True,
                separate_cfg_input: bool = False,
                use_input_image_size_as_output: bool=False,
                llm_type=None, llm_processor=None, input_llm_images=None, llm_user_input=None, pure_text=None,
                ) -> Dict:

        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]
        
        if llm_user_input is not None and isinstance(llm_user_input, str):
            llm_user_input = [llm_user_input]
        
        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction
            
            if llm_user_input is not None:
                short_instruction = llm_user_input[i]
            else:
                short_instruction = None
            mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images, llm_type=llm_type, llm_processor=llm_processor, input_llm_images=input_llm_images, short_instruction=short_instruction, pure_text=pure_text)

        
            neg_mllm_input, img_cfg_mllm_input = None, None
            # TODO: input_llm_images For text cfg, should it be entered? Not entering will have a negative effect.
            # neg_mllm_input = self.process_multi_modal_prompt(negative_prompt, None, llm_type=llm_type, llm_processor=llm_processor, input_llm_images=input_llm_images)
            neg_mllm_input = self.process_multi_modal_prompt(negative_prompt, None, llm_type=llm_type, llm_processor=llm_processor)
            if use_img_cfg:
                if cur_input_images is not None and len(cur_input_images) >= 1:
                    img_cfg_prompt = [f"<img><|image_{i+1}|></img>" for i in range(len(cur_input_images))]
                    img_cfg_mllm_input = self.process_multi_modal_prompt(" ".join(img_cfg_prompt), cur_input_images, llm_type=llm_type, llm_processor=llm_processor)
                else:
                    img_cfg_mllm_input = neg_mllm_input

            if use_input_image_size_as_output:
                input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, [mllm_input['pixel_values'][0].size(-2), mllm_input['pixel_values'][0].size(-1)]))
            else:
                input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, [height, width]))

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
            temp_position = [0]*(text_length-temp_l) + [i for i in range(temp_l+img_length+1)] # we add a time embedding into the sequence, so add one more token
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)

    def create_llm_vae_position(self, llm_vae_attention_mask, llm_2d_attention_mask, num_tokens_for_output_images, llm_image_sizes):
        position_ids = []
        text_length = llm_vae_attention_mask.size(-1)
        # img_length = max(num_tokens_for_output_images)  
        for batch_idx, mask in enumerate(llm_vae_attention_mask):
            temp_l = torch.sum(llm_2d_attention_mask[batch_idx])
            # temp_position = [0]*(text_length-temp_l) + [i for i in range(temp_l+img_length+1)] # we add a time embedding into the sequence, so add one more token
            temp_position = [0]*(text_length-temp_l) + [i for i in range(temp_l)] # only condition for mllm like qwen
            if batch_idx in llm_image_sizes:
                for vae_img_id in llm_image_sizes[batch_idx]:
                    temp_position[vae_img_id[0]:vae_img_id[1]] = [0]*(vae_img_id[1] - vae_img_id[0])
                    temp_back = cp.deepcopy(temp_position[vae_img_id[1]:])
                    temp_position[vae_img_id[1]:] = [pos_id - (vae_img_id[1] - vae_img_id[0]) for pos_id in temp_back]
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)
    
    def create_mask(self, attention_mask, num_tokens_for_output_images):
        extended_mask = []
        padding_images = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1 # we add a time embedding into the sequence, so add one more token
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = text_length - temp_l

            temp_mask = torch.tril(torch.ones(size=(temp_l+1, temp_l+1)))

            image_mask = torch.zeros(size=(temp_l+1, img_length))
            temp_mask = torch.cat([temp_mask, image_mask], dim=-1)

            image_mask = torch.ones(size=(img_length, temp_l+img_length+1))
            temp_mask = torch.cat([temp_mask, image_mask], dim=0)

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l+1+img_length, pad_l))
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
    
    def pad_input_ids(self, input_ids, image_sizes, llm_mode=False):
        pad_token_id = self.pad_token_id if not llm_mode else self.llm_pad_token_id  # 151642 <|endoftext|> in qwen2.5vl
        max_l = max([len(x) for x in input_ids])
        padded_ids = []
        attention_mask = []
        if llm_mode:
            vae_attention_mask = []
        new_image_sizes = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1]*max_l)
                if llm_mode:
                    vae_attention_mask.append([1]*max_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0]*pad_l+[1]*temp_l)
                if llm_mode:
                    vae_attention_mask.append([0]*pad_l+[1]*temp_l)
                padded_ids.append([pad_token_id]*pad_l+temp_ids)
            
            if i in image_sizes:
                new_inx = []
                for old_inx in image_sizes[i]:
                    new_inx.append([x+pad_l for x in old_inx])
                    if llm_mode:
                        vae_attention_mask[i][old_inx[0] + pad_l: old_inx[1] + pad_l] = [0] * (old_inx[1] - old_inx[0])
                image_sizes[i] = new_inx

        if llm_mode:
            return torch.LongTensor(padded_ids), torch.LongTensor(vae_attention_mask), torch.LongTensor(attention_mask), image_sizes
        else:
            return torch.LongTensor(padded_ids), None, torch.LongTensor(attention_mask), image_sizes

    def process_mllm_input(self, mllm_inputs, target_img_size):
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0]*img_size[1]//16//16)

        pixel_values, image_sizes, llm_image_sizes = [], {}, {}
        b_inx = 0
        for x in mllm_inputs:
            if x['pixel_values'] is not None:
                pixel_values.extend(x['pixel_values'])
                if x['llm_img_inx'] is not None:
                    for size, llm_size in zip(x['image_sizes'], x['llm_img_inx']):
                        if b_inx not in image_sizes:
                            image_sizes[b_inx] = [size]
                            llm_image_sizes[b_inx] = [llm_size]
                        else:
                            image_sizes[b_inx].append(size)
                            llm_image_sizes[b_inx].append(llm_size)
                else:
                    for size in x['image_sizes']:
                        if b_inx not in image_sizes:
                            image_sizes[b_inx] = [size]
                        else:
                            image_sizes[b_inx].append(size)
            b_inx += 1     
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        
        input_ids = [x['input_ids'] for x in mllm_inputs]
        padded_input_ids, _, attention_mask, image_sizes = self.pad_input_ids(input_ids, image_sizes, llm_mode=False)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output_images)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)

        llm_input_ids = [x['llm_input_ids'] for x in mllm_inputs]
        if llm_input_ids[0] is not None:
            llm_inputs = [x['llm_inputs'] for x in mllm_inputs]
            llm_padded_input_ids, llm_vae_attention_mask, llm_2d_attention_mask, llm_image_sizes = self.pad_input_ids(llm_input_ids, llm_image_sizes, llm_mode=True)
            llm_position_ids = self.create_position(llm_2d_attention_mask, num_tokens_for_output_images)
            llm_vae_position_ids = self.create_llm_vae_position(llm_vae_attention_mask, llm_2d_attention_mask, num_tokens_for_output_images, llm_image_sizes)
            llm_attention_mask, padding_images = self.create_mask(llm_2d_attention_mask, num_tokens_for_output_images)
            llm_attention_mask = self.adjust_attention_for_input_images(llm_attention_mask, llm_image_sizes) # Do: vae feature mask convert into llm
        else:
            llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids = None, None, None, None, None, None, None, None

        return padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes, llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids
    
    
    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]

        
        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size + target_img_size
        else:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs
            target_img_size = target_img_size + target_img_size


        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes, llm_vae_attention_mask = self.process_mllm_input(mllm_inputs, target_img_size)

        data = {"input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "input_pixel_values": all_pixel_values,
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        }
        return data


class OmniGenSeparateCollator(OmniGenCollator):
    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]
        
        all_padded_input_ids, all_attention_mask, all_position_ids, all_pixel_values, all_image_sizes, all_padding_images = [], [], [], [], [], []
        all_llm_padded_input_ids, all_llm_2d_attention_mask, all_llm_attention_mask, all_llm_position_ids, all_llm_inputs, all_llm_image_sizes, all_llm_vae_attention_mask, all_llm_vae_position_ids = [], [], [], [], [], [], [], []

        padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes,\
             llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids = self.process_mllm_input(mllm_inputs, target_img_size)
        all_padded_input_ids.append(padded_input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_padding_images.append(padding_images)
        all_llm_padded_input_ids.append(llm_padded_input_ids)
        all_llm_2d_attention_mask.append(llm_2d_attention_mask)
        all_llm_attention_mask.append(llm_attention_mask)
        all_llm_position_ids.append(llm_position_ids)
        all_llm_inputs.append(llm_inputs)
        all_llm_image_sizes.append(llm_image_sizes)
        all_llm_vae_attention_mask.append(llm_vae_attention_mask)
        all_llm_vae_position_ids.append(llm_vae_position_ids)

        if cfg_mllm_inputs[0] is not None:
            padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes,\
                 llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids = self.process_mllm_input(cfg_mllm_inputs, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
            all_llm_padded_input_ids.append(llm_padded_input_ids)
            all_llm_2d_attention_mask.append(llm_2d_attention_mask)
            all_llm_attention_mask.append(llm_attention_mask)
            all_llm_position_ids.append(llm_position_ids)
            all_llm_inputs.append(llm_inputs)
            all_llm_image_sizes.append(llm_image_sizes)
            all_llm_vae_attention_mask.append(llm_vae_attention_mask)
            all_llm_vae_position_ids.append(llm_vae_position_ids)

        if img_cfg_mllm_input[0] is not None:
            padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes,\
                 llm_padded_input_ids, llm_2d_attention_mask, llm_attention_mask, llm_position_ids, llm_inputs, llm_image_sizes, llm_vae_attention_mask, llm_vae_position_ids = self.process_mllm_input(img_cfg_mllm_input, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
            all_llm_padded_input_ids.append(llm_padded_input_ids)
            all_llm_2d_attention_mask.append(llm_2d_attention_mask)
            all_llm_attention_mask.append(llm_attention_mask)
            all_llm_position_ids.append(llm_position_ids)
            all_llm_inputs.append(llm_inputs)
            all_llm_image_sizes.append(llm_image_sizes)
            all_llm_vae_attention_mask.append(llm_vae_attention_mask)
            all_llm_vae_position_ids.append(llm_vae_position_ids)

        data = {"input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids,
        "input_pixel_values": all_pixel_values,
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        "llm_padded_input_ids": all_llm_padded_input_ids,
        "llm_2d_attention_mask": all_llm_2d_attention_mask,
        "llm_attention_mask": all_llm_attention_mask,
        "llm_position_ids": all_llm_position_ids,
        "llm_inputs": all_llm_inputs,
        "llm_image_sizes": all_llm_image_sizes,
        "llm_vae_attention_mask": all_llm_vae_attention_mask,
        "llm_vae_position_ids": all_llm_vae_position_ids,
        }
        return data
