import torch
from .data import DatasetFromJson

class ConcateNate(torch.utils.data.Dataset):
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
        t2i_ratio: float = 0.0,
        t2i_json: str = '/group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset',
        t2i_file_name: str = 'tar.gz',
        t2i_keys=['png', 'txt'],
        llm_type: str = 'qwen2.5vl',
        x2i_length: int = None,
        t2i_length: int = None,
        use_longshort_t2i: bool = False,
    ):
        self.data_1 = 
        self.t2i_dataset_length = int(len(self.t2i_webdataset) * self.t2i_ratio)
        self.llm_type = llm_type
        self.llm_processor = llm_processor
        print(f'X2I data number: {self.x2i_length}')
        print(f'T2I data number: {self.t2i_dataset_length}')

    def process_image(self, image_file):
        # if self.image_path is not None:
            # image_file = os.path.join(self.image_path, image_file)
        image = Image.open(image_file).convert('RGB')
        return self.image_transform(image)

    def get_example(self, index):
        example = self.data[index]
        
        instruction, input_images, output_image = example['instruction'], example['input_images'], example['output_image']
        qwen_instruction = example.get('instruction_qwen', None)
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

        mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images, llm_type=self.llm_type, llm_processor=self.llm_processor, short_instruction=short_instruction, input_llm_images=input_llm_images)

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

    def __getitem__(self, index):
        if index >= self.x2i_length:
            return self.t2i_webdataset.get_data_info(index)
        else:
            if self.x2i_length < len(self.data):
                index = random.randint(0, len(self.data) - 1)
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
        return self.x2i_length + self.t2i_dataset_length
