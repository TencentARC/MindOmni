import json
import jsonlines
import os
from omegaconf import OmegaConf


def load_yaml_dataset(datasets, key_list, debug_json_item_num=None):
    list_data_all = []
    json_info_all = []
    for dataset in datasets:
        json_path = dataset['json_path']
        image_dir = dataset.get('image_dir', None)

        if image_dir is not None:
            # load json
            # print(f"Loading {json_path} and add {image_dir}")
            list_data = load_annotation(json_path, is_replace_image_file=True, image_dir=image_dir, key_list=key_list)
        else:
            list_data = load_annotation(json_path, is_replace_image_file=False, image_dir=image_dir, key_list=key_list)
        if debug_json_item_num is not None:
            list_data = list_data[0][:debug_json_item_num]
        else:
            list_data = list_data[0]
        json_info_all.append((json_path, len(list_data)))
        list_data_all.extend(list_data)
    return list_data_all, json_info_all


def load_annotation(data_path, is_replace_image_file=False, image_dir=None, key_list=["input_images"], debug_json_item_num=None):
    json_info_all = None
    if os.path.splitext(data_path)[1] == ".json":
        list_data_dict = json.load(open(data_path, "r"))
        # replace
        if is_replace_image_file and image_dir is not None:
            for sample in list_data_dict:
                for key in key_list:
                    if key in sample:
                        if isinstance(sample[key], list):
                            sample[key] = [os.path.join(image_dir, x) for x in sample[key]]
                        else:
                            sample[key] = os.path.join(image_dir, sample[key])
    elif os.path.splitext(data_path)[1] == ".jsonl":
        with jsonlines.open(data_path, 'r') as reader:
            list_data_dict = [obj for obj in reader]
        # replace
        if is_replace_image_file and image_dir is not None:
            for sample in list_data_dict:
                for key in key_list:
                    if key in sample:
                        if isinstance(sample[key], list):
                            sample[key] = [os.path.join(image_dir, x) for x in sample[key]]
                        else:
                            sample[key] = os.path.join(image_dir, sample[key])
    elif os.path.splitext(data_path)[1] == ".yaml":
        config = OmegaConf.load(data_path)
        datasets = config.datasets
        list_data_dict, json_info_all = load_yaml_dataset(datasets, key_list=key_list, debug_json_item_num=debug_json_item_num)
    else:
        raise NotImplementedError
    return list_data_dict, json_info_all
