# Copyright (c) Lin Song. All rights reserved.
import sys
import io
from glob import glob

import webdataset as wds
from PIL import Image
import os
from torchvision import transforms
import torch
import json
from OmniGen import OmniGenProcessor
import random


class X2IWebDataset(torch.utils.data.Dataset):
    """JourneyDB WebDataset dataset."""

    def __init__(
            self,
            image_transform,
            llm_processor,
            processer: OmniGenProcessor,
            data_root='/group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset',
            epoch_length: int = 4134263,
            file_name='tar.gz',
            keys=['png', 'txt'],
            llm_type=None,  # 'qwen2.5vl'
            max_input_length_limit: int = 18000,
            condition_dropout_prob: float = 0.1,
            keep_raw_resolution: bool = True,
        ) -> None:
        """Initialize the dataset."""
        self.data_root = data_root
        self.image_transform = image_transform
        self.processer = processer
        self.condition_dropout_prob = condition_dropout_prob
        self.max_input_length_limit = max_input_length_limit
        self.keep_raw_resolution = keep_raw_resolution
        self.llm_type = llm_type
        self.llm_processor = llm_processor

        self.epoch_length = epoch_length
        self.keys = keys
        self.file_name = file_name
        self.full_init()

    def get_data_info(self, index: int) -> dict:
        """Get data info from the dataset."""
        image, caption = next(self.dataset)
        caption = caption.decode('utf-8')
        image = Image.open(io.BytesIO(image)).convert('RGB')
        # import ipdb; ipdb.set_trace()

        instruction = caption
        if random.random() < self.condition_dropout_prob:
            instruction = '<cfg>'
        input_images = None

        mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images, llm_type=self.llm_type, llm_processor=self.llm_processor)

        output_image = self.image_transform(image)

        return (mllm_input, output_image)

    def full_init(self) -> None:
        """Fully initialize the dataset."""
        # tar_files = glob(self.data_root)
        if self.file_name == 'tar':
            tar_files = glob(os.path.join(self.data_root, "*.tar"), recursive=True)
        elif self.file_name == 'tar.gz':
            tar_files = glob(os.path.join(self.data_root, "*.tar.gz"), recursive=True)
        self.dataset = wds.WebDataset(
            tar_files,
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.shardlists.split_by_node)

        self.dataset = self.dataset.to_tuple(self.keys[0], self.keys[1])
        self.dataset = self.dataset.with_epoch(sys.maxsize)
        self.dataset = iter(self.dataset)

    def __getitem__(self, index):
        return self.get_data_info(index)

    def __len__(self) -> int:
        assert self.epoch_length is not None
        return self.epoch_length


if __name__ == "__main__":
    from OmniGen.utils import center_crop_arr
    from transformers import AutoProcessor
    processor = OmniGenProcessor.from_pretrained('/group/40043/yichengxiao/huggingface/model/OmniGen-v1')
    llm_processor = AutoProcessor.from_pretrained('/group/40043/yichengxiao/huggingface/model/Qwen2.5-VL-7B-Instruct')
    crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = X2IWebDataset(
        image_transform=image_transform,
        processer=processor,
        llm_processor=llm_processor,
        # data_root='/group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset',
        data_root='/group/40043/public_datasets/BLIP3o-text',
        # data_root='/group/40121/public_datasets/LeX-10K/data_webdataset',
        keys = ['jpg', 'txt'],
        # keys=['image', 'caption'],
        # keys = ['jpg', 'json'],
        # file_name='tar.gz',
        file_name='tar',
    )

    def save_single_image():
        data = dataset[0]
        img = data['video'][0]
        caption = data['gt_caption']
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        original_image = img * std + mean  # 反标准化

        # 将张量剪裁到合法范围 [0, 1]
        original_image = torch.clamp(original_image, 0, 1)

        # 转换为 PIL 图像
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(original_image)
        pil_image.save("test_debug.jpg")
        print(caption)

    save_single_image()
    # img = dataset[0]['video'][0]
    import ipdb; ipdb.set_trace()
    print(1)
