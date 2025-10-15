import torch
from OmniGen import OmniGenPipeline
import os
from PIL import Image
import re
from qwen_vl_utils import process_vision_info
# from xyc_utils.resize_image import crop_arr
from ..utils import crop_arr, crop_by_max_pixels


def build_message(user_input, think_content, qwen_prompt, llm_processor, model_llm, use_template, logger):
    system_prompt = (
        "You are a helpful assistant. When the user requests an image, the assistant "
        "first thinks about the reasoning process in the mind and then provides the user with concise prompt as the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>."
    )

    if not use_template:
        processor = llm_processor
        model = model_llm

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

        # Inference: Generation of the output
        # generated_ids = model.generate(**inputs, do_sample=True, temperature=1, max_new_tokens=512)
        generated_ids = model.generate(**inputs, do_sample=False, temperature=None, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if '</answer>' in output_text[0]:
            prompt_ = re.split(r"</answer>", output_text[0])[0]
        else:
            prompt_ = output_text[0]
        # prompt_ = output_text[0]

        assistant_content = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": prompt_},
                ],
            }
        ]

        logger.info(prompt_)
        messages += assistant_content
        return messages

    else:
        prompt_organize = f"<think> {think_content} </think><answer> {qwen_prompt}"
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
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": prompt_organize},
                ],
            }
        ]
        return messages


def build_message_edit(edit_prompt, input_llm_images, think_content, max_input_image_size):
    system_prompt = (
        "You are a helpful assistant. When the user requests an image, the assistant "
        "first thinks about the reasoning process in the mind and then generates high-quality images based on user instructions and reasoning content. "
        "The reasoning process is enclosed within <think> </think> tags, i.e., "
        "<think> reasoning process here </think>."
    )           # training for omniedit dataset
    if think_content is None:
        prompt_organize = "<|BOI|>"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate an image according to the following instructions\n"},
                    {"type": "text", "text": edit_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": prompt_organize},
                ],
            }
        ]
        image_tags = re.findall(r'<\|image_\d+\|>', messages[0]['content'][1]['text'])
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        pattern = r"<img><\|image_\d+\|></img>"
        prompt_chunks = [chunk for chunk in re.split(pattern, messages[0]['content'][1]['text'])]
        assert len(prompt_chunks) == len(input_llm_images) + 1
        # assert len(prompt_chunks) == len(input_llm_images)  # 用于只训omniedit mllm，使用了target作为input image
        new_content = []
        for idx, per_prompt in enumerate(prompt_chunks):
            if idx != len(prompt_chunks) - 1:
                item_text = {"type": "text", "text": per_prompt}
                # resized_height, resized_width = input_images_shape[image_ids[idx] - 1]
                image_path = input_llm_images[image_ids[idx] - 1]
                # resized_width, resized_height = crop_arr(Image.open(image_path), max_image_size=max_input_image_size).size
                resized_width, resized_height = crop_by_max_pixels(Image.open(image_path), max_input_image_size * max_input_image_size).size
                item_vit = {"type": "image", "image": image_path, "resized_height": resized_height, "resized_width": resized_width}
                # item_vit = {"type": "image", "image": image_path}
                item_tag = {"type": "text", "text": f"<img>{image_tags[idx]}</img>"}
                new_content.append(item_text)
                new_content.append(item_vit)
                new_content.append(item_tag)
            else:
                item_text = {"type": "text", "text": per_prompt}
                new_content.append(item_text)
        messages[0]['content'] = messages[0]['content'][:1] + new_content
    else:
        prompt_organize = f"<think> {think_content} </think><|BOI|>"  # <BOI> used for generation imaga
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
                    {"type": "text", "text": edit_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": prompt_organize},
                ],
            }
        ]
        new_omnigen_text = "<img><|image_1|></img> " + think_content    # 只用于训omniedit数据时带think的情况，用think内容来蒸馏
        image_tags = re.findall(r'<\|image_\d+\|>', messages[1]['content'][1]['text'])
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        pattern = r"<img><\|image_\d+\|></img>"
        prompt_chunks = [chunk for chunk in re.split(pattern, messages[1]['content'][1]['text'])]
        assert len(prompt_chunks) == len(input_llm_images) + 1
        # assert len(prompt_chunks) == len(input_llm_images)  # 用于只训omniedit mllm，使用了target作为input image
        new_content = []
        for idx, per_prompt in enumerate(prompt_chunks):
            if idx != len(prompt_chunks) - 1:
                item_text = {"type": "text", "text": per_prompt}
                # resized_height, resized_width = input_images_shape[image_ids[idx] - 1]
                image_path = input_llm_images[image_ids[idx] - 1]
                # resized_width, resized_height = crop_arr(Image.open(image_path), max_image_size=max_input_image_size).size
                resized_width, resized_height = crop_by_max_pixels(Image.open(image_path), max_input_image_size * max_input_image_size).size
                item_vit = {"type": "image", "image": image_path, "resized_height": resized_height, "resized_width": resized_width}
                # item_vit = {"type": "image", "image": image_path}
                item_tag = {"type": "text", "text": f"<img>{image_tags[idx]}</img>"}
                new_content.append(item_text)
                new_content.append(item_vit)
                new_content.append(item_tag)
            else:
                item_text = {"type": "text", "text": per_prompt}
                new_content.append(item_text)
        messages[1]['content'] = messages[1]['content'][:1] + new_content

    return messages


def construct_debug_prompt_list():
    data = []
    import jsonlines
    # validate_file = 'data/mindomni_edit_validate_2.jsonl'
    validate_file = 'data/mindomni_edit_validate.jsonl'
    with open(validate_file, 'r') as file:
        for obj in jsonlines.Reader(file):
            if len(obj['input_images']) < 1:
                item = {
                    'prompt': obj['instruction'],
                    'output_image': obj['output_image']
                }
            else:
                item = {
                    'edit_prompt': obj['instruction'],
                    'input_images': obj['input_images'],
                    'output_image': obj['output_image']
                }
            data.append(item)
    return data

@torch.no_grad()
def validate_func(accelerator, model, vae, processor, model_llm, llm_processor, logger, base_dir, step, device, val_img_size):
    base_dir = os.path.join(base_dir, 'val_imgs')
    os.makedirs(base_dir, exist_ok=True)
    model.eval()

    model_llm_training_flag = False
    if model_llm.training:
        model_llm_training_flag = True
        model_llm.eval()

    pipe = OmniGenPipeline(model=accelerator.unwrap_model(model), vae=vae, processor=processor,
                           model_llm=accelerator.unwrap_model(model_llm), llm_processor=llm_processor, device=device)
    logger.info("Generating images...")

    prompt_list = [
        # "Neon words 'UST' areflashing in the prosperous futurecity, the sense of science andtechnology, quality details, hyperrealistic, high definition, 8K, photo,best quality, high quality.",
        # "A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light. She is dressed in a cozy cream sweater. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug.The background is a serene indoor setting, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance.",
        # "A vintage camera placed on the ground, ejecting a swirling cloud of Polaroid-style photographs into the air. The photos, showing landscapes, wildlife, and travel scenes, seem to defy gravity, floating upward in a vortex of motion. The camera emits a glowing, smoky light from within, enhancing the magical, surreal atmosphere. The dark background contrasts with the illuminated photos and camera, creating a dreamlike, nostalgic scene filled with vibrant colors and dynamic movement. Scattered photos are visible on the ground, further contributing to the idea of an explosion of captured memories.",
        # "Animal associated with having (2 + 7) lives.",
        # "A vibrant still life of a shiny red apple resting on a smooth white wooden table in a rustic kitchen, bathed in soft golden morning light streaming through a window, with a faint shadow cast across the surface and a blurred background of wooden shelves filled with jars.",
        # "一只可爱迷人的小狐狸，长着大大的棕色眼睛，背景中是迷人的秋叶，（它看起来）仿佛永生般，拥有毛茸茸、亮闪闪的鬃毛，周围还有花瓣，充满奇幻色彩，采用虚幻引擎 5 和辛烷渲染器打造，细节高度精细，画面逼真写实，具有电影质感，色彩自然。",
        # "a young woman, looks like mix of Lana Del Rey andgrimes, flowing cool colored hair, marbled, iridescent,shoujo manga, pre-raphaelite, k-pop, gilded, pearl,spun silk, clouds, ghost, glowing jellyfish, billowinggossamer cloth, Alexander McQueen, handmade lace,floral embroidery, snakeskin, dramatic lighting.",
        # {"prompt": "A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light. She is dressed in a cozy cream sweater. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug.The background is a serene indoor setting, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance."},
        # {"prompt": "Convert the image <img><|image_1|></img> to watercolor style.", "input_images": ['img5.png']},
        # {"user_input": "<img><|image_1|></img> Please replace food contains most vitamin with a pineapple.", "prompt": "Please replace the apple on the right in the image with a pineapple.", "input_images": ['img15.png']},
        # {"prompt": "Neon words 'UST' are flashing in the prosperous futurecity, the sense of science andtechnology, quality details, hyperrealistic, high definition, 8K, photo,best quality, high quality."},
        # {"prompt": "<img><|image_1|></img> Remove the piece in c1 cell and add to c2 cell.", "input_images": ['data/Chess_Data/chess_outputs/1_Kc2_c1c2.png']},
        # {"prompt": "<img><|image_1|></img> Remove the piece in d6 cell and add to c5 cell.", "input_images": ['data/Chess_Data/chess_outputs/3827_Kc5_d6c5.png']},
        # {"prompt": "<img><|image_1|></img> Remove the piece in f4 cell and add to d4 cell.", "input_images": ['data/Chess_Data/chess_outputs/227_Rd4_f4d4.png']},
        # {"prompt": "<img><|image_1|></img> Add the piece 'O' in B3 cell.", "input_images": ['data/Chess_Data/tictactoe_outputs/2_X_B3.png']},
        # {"prompt": "<img><|image_1|></img> Add the piece 'O' in A1 cell.", "input_images": ['data/Chess_Data/tictactoe_outputs/2_X_B3.png']},
        # {"prompt": "<img><|image_1|></img> Add the piece 'X' in B3 cell.", "input_images": ['data/Chess_Data/tictactoe_outputs/2_X_B3.png']},
        # {"user_input": "Generate an image about the animal associated with having (2 + 7) lives.", "prompt": "Create an image of a cat, symbolizing the mythical creature with nine lives, depicted in a whimsical, cartoonish style with vibrant colors and a playful, cheerful mood. The cat should have a soft, fluffy texture and be surrounded by a glowing aura, emphasizing its legendary resilience."},
        # {"user_input": "Generate an image about traditional decoration for the Mexican Day of the Deads.", "prompt": "Create an image of ofrendas, elaborate altars decorated with marigolds, candles, and offerings."},
        # {"user_input": "Generate an image about symbolic animal associated with the Chinese New Year.", "prompt": "Create an image of the dragon, a mythical creature representing power and good fortune."},
        # {"prompt": "Neon words 'UST' areflashing in the prosperous futurecity, the sense of science andtechnology, quality details, hyperrealistic, high definition, 8K, photo,best quality, high quality."},
        # # gedit benchmark
        {"edit_prompt": "<img><|image_1|></img> Adjust the background to a beach.", "input_images": ["img_sources/bg_57288ae252f43831390e2121a84b1780.png"]},
        {"edit_prompt": "<img><|image_1|></img> Change the zebra’s material to concrete.", "input_images": ["img_sources/material_5098e702ebab84dc41c1ec86a937bfb2.png"]},
        {"edit_prompt": "<img><|image_1|></img> Make the person in the image wave.", "input_images": ["img_sources/motion_fc228a38f175cad001bc8a409c76e63b.png"]},
        {"edit_prompt": "<img><|image_1|></img> Enhance my nose.", "input_images": ["img_sources/pshuman_697678d3816a0fcfc357a108ae47955a.png"]},
        {"edit_prompt": "<img><|image_1|></img> Convert this image into an anime style.", "input_images": ["img_sources/style_599dbcd5dd042cec90da287aa11414ce.png"]},
        {"edit_prompt": "<img><|image_1|></img> Add two small dogs sitting face-to-face in the foreground.", "input_images": ["img_sources/add_bcb9d7a80eaf8a5f630cc78b6bce0b6c.png"]},
        {"edit_prompt": "<img><|image_1|></img> Enhance this image by removing the distant power lines while maintaining a realistic style.", "input_images": ["img_sources/remove_cd5e2a6dd0f762849943fede284c4516.png"]},
        {"edit_prompt": "<img><|image_1|></img> Swap the bouquet in the woman’s hand for a bottle of whiskey.", "input_images": ["img_sources/replace_19e7dd610e2151dd4576490c7ece040f.png"]},
        {"edit_prompt": "<img><|image_1|></img> Replace the text 'PIZZA' with 'PLAZA'.", "input_images": ["img_sources/text_853bc02c90873ac8838e53ee11fa5ec3.png"]},
        {"edit_prompt": "<img><|image_1|></img> Apply an HDR filter to brighten the image.", "input_images": ["img_sources/tone_8168e81061f790fb34c9f4c81ed34d90.png"]},
        # # #### exp70 val case begin
        {"prompt": "A cat holds a sign that says 'MindOmni'."},
        {"prompt": "A menu board in a café showing three items: “Espresso”, “Latte”, and “Cappuccino”."},
        # {"prompt": "a photo of a blue pizza and a yellow baseball glove."},
        # {"prompt": "a photo of a tennis racket right of a spoon."},
        # case from uniworld valid:
        # {"edit_prompt": "<img><|image_1|></img> Render an image where fine details and textures are filled in based on the provided canny lines, influenced by 'white and black dogs on snow covered ground during daytime'.", "input_images": ["img_sources/canny.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Generate a Canny edge map for this image.", "input_images": ["img_sources/canny_image.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Create a pose map using OpenPose.", "input_images": ["img_sources/pose_image.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Create a person image that conforms to the input pose, with realistic anatomy and appearance related to 'Two individuals sit on a wooden bench in a park, with one person stretching their arms above their head and the other engrossed in their mobile device.'.", "input_images": ["img_sources/pose.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Switch the product's color from black to white, making sure the transition is crisp and clear.", "input_images": ["img_sources/nike_src.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Convert an image to Ghibli style.", "input_images": ["img_sources/bus.png"]},
        # {"edit_prompt": "<img><|image_1|></img> Extract the ny 94 printed cotton-jersey sweatpants from the person, ensuring the image only displays the item without any background distractions.", "input_images": ["img_sources/extract_src.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Integrate the striped cotton sweater into the person's overall look, making it appear natural and stylish.", "input_images": ["img_sources/extract_dst.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> replace motorcycle located in the lower center region of the image with a black bicycle.", "input_images": ["img_sources/replace_src.png"]},
        # {"edit_prompt": "<img><|image_1|></img> Segment the giraffe from the background.", "input_images": ["img_sources/seg_src.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Please detect the vase accurately.", "input_images": ["img_sources/det_src.jpg"]},
        # {"edit_prompt": "<img><|image_1|></img> Convert it into a Studio Ghibli art style.", "input_images": ["img_sources/task_style_116833_src.png"]},
        # case custom by xyc:
        # {"edit_prompt": "<img><|image_1|></img> Remove chair", "think_content": "Remove a modern black chair near the window in the background.", "input_images": ["img_sources/task_obj_remove_216690_src.png"]},
        # {"edit_prompt": "The flower <img><|image_1|></img> is placed in the vase which is in the middle of <img><|image_2|></img> on a wooden table of a living room.", "input_images": ["img_sources/rose.jpg", "img_sources/vase.jpg"]},
        # {"edit_prompt": "Please let the person in <img><|image_1|></img> hold the toy from <img><|image_2|></img> in a parking lot.", "input_images": ["img_sources/000365954.jpg", "img_sources/04.jpg"]},
        # {"edit_prompt": "Two woman are raising fried chicken legs in a bar. A woman is <img><|image_1|></img>. Another woman is <img><|image_2|></img>.", "input_images": ["img_sources/Amanda.jpg", "img_sources/mckenna.jpg"]},
        # case for in-context generation:
        {"edit_prompt": "Adapt image <img><|image_1|></img> to fit the aesthetic of image <img><|image_2|></img>.", "input_images": ["img_sources/00182555_target.jpg", "img_sources/00182555_InstantStyle_ref_1.jpg"]},
        {"edit_prompt": "Create a wedding figure based on the girl in <img><|image_1|></img> and the man in <img><|image_2|></img>. Set the background as a wedding hall, with the man dressed in a suit and the girl in a white wedding dress. Ensure that the original faces remain unchanged and are accurately preserved. The man should adopt a realistic style, whereas the girl should maintain their classic anime style.", "input_images": ["img_sources/1_20241127203215.png", "img_sources/000050281.jpg"]},
        # {"edit_prompt": "Make the girl in <img><|image_1|></img> pray in the second image <img><|image_2|></img>.", "input_images": ["img_sources/000440817.jpg", "img_sources/000119733.jpg"]},
        {"edit_prompt": "Replace the dog in <img><|image_1|></img> with the animal in <img><|image_2|></img>.", "input_images": ["img_sources/dog_woman.png", "img_sources/duck.png"]},
        # # #### exp70 val case end
        # {"user_input": "A highly technical and fast-paced sport in China, often showcasing incredible agility and precision."},
        # {"user_input": "Symbolic animal associated with the Chinese New Year."},
        # {"user_input": "A highly technical and fast-paced sport in China, often showcasing incredible agility and precision", "think_content": "The prompt describes a sport in China characterized by technical skill, speed, agility, and precision. Among sports popular in China, table tennis (ping pong) stands out as highly technical and fast-paced, well-known for rallies that exhibit quick reflexes and precise control. To make the subject obvious to the viewer, key visual elements like the table tennis ball and paddle should be prominent in the image.", "instruction_qwen": 'The image should depict a high-energy table tennis match. Include a table tennis table with a net, two players with intense expressions, each holding a paddle, and a blurred ball to indicate speed. The focus could be on one player expertly returning a fast shot, highlighting their agility and technical mastery. Ensure the paddle and ball are clearly visible and central to the composition, clearly identifying the sport as table tennis.'},
        # {"user_input": "Symbolic animal associated with the Chinese New Year.", "think_content": "The prompt asks for an image of an animal symbolically linked to the Chinese New Year. Each year in the Chinese Zodiac is associated with a specific animal, such as the dragon, which is not only part of the zodiac but also culturally symbolizes power, luck, and prosperity. The explanation highlights the dragon, one of the most revered creatures in Chinese tradition, especially prominent during festivals like the Chinese New Year.", "instruction_qwen": 'The symbolic animal for the Chinese New Year often varies according to the yearly cycle of the Chinese Zodiac, which includes twelve animals. However, the dragon stands out as a culturally significant symbol apart from the zodiac cycle. The dragon is deeply associated with the Chinese New Year celebrations, representing power, strength, and good fortune. Dragon dances are a traditional part of the festivities, believed to bring luck for the coming year.'},
        # {"user_input": "An image of multiple apples, the quantity of apples is the solution of 'x^2 + 2 = 11'."},
        # {"user_input": "An image of (2 * 4 - 4) apples."},
        # {"prompt": "<img><|image_1|></img> Add a pair of sunglasses to the cat in the image.", "input_images": ['img_sources/img2.png']},
        # {"prompt": "<img><|image_1|></img> Find all windowpanes in the photo and outline them in blue.", "input_images": ['img_sources/img17.png']},
        # case for debug uniworld
        # {"edit_prompt": "<img><|image_1|></img> Record the boundaries of broccoli occurrences", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_seg_box-448k/images/OpenDataLab___COCO_2017/raw/Images/train2017/000000000009.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_seg_box-448k/images/drawed_coco/train_2017/bbox/000000000009_broccoli.jpg"},
        # {"edit_prompt": "Synthesize a Canny edge rendition of this image. <img><|image_1|></img>", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_caption_canny-236k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_caption_canny-236k/images/00000001.jpg"},
        # {"edit_prompt": "Predict depth affected by occlusion and overlap. <img><|image_1|></img>", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_caption_depth-236k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_caption_depth-236k/images/00000001.jpg"},
        # {"edit_prompt": "<img><|image_1|></img> Highlight broccoli with precise segmentation.", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_seg_box-448k/images/OpenDataLab___COCO_2017/raw/Images/train2017/000000000009.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_seg_box-448k/images/drawed_coco/train_2017/seg_opacity_0.8/000000000009_broccoli.jpg"},
        # {"edit_prompt": "<img><|image_1|></img> Produce a clean pass normal map from the photo.", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_caption_normal-236k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_caption_normal-236k/images/00000001.jpg"},
        # {"edit_prompt": "<img><|image_1|></img> Render a gestural pencil impression of this image.", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_caption_sketch-236k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_caption_sketch-236k/images/00000001.jpg"},
        # {"edit_prompt": "<img><|image_1|></img> Construct a skeletal keypoint map highlighting posture.", "input_images": ["/group/40075/public_datasets/UniWorld/data/coco2017_caption_openpose-62k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/coco2017_caption_openpose-62k/images/00000001.jpg"},
        # {"edit_prompt": "Compute pixel-level Canny edge representation. <img><|image_1|></img>", "input_images": ["/group/40075/public_datasets/UniWorld/data/mscoco-controlnet-canny-less-colors-236k/images/00000000.jpg"], "output_image": "/group/40075/public_datasets/UniWorld/data/mscoco-controlnet-canny-less-colors-236k/images/00000001.jpg"},
    ]

    # use for debug
    # prompt_list = construct_debug_prompt_list()

    images_result = []
    output_image_gt_result = []
    input_image_list = []
    for idx, prompt in enumerate(prompt_list):
        prompt_ = prompt.get('prompt', '')
        input_images = prompt.get('input_images', None)
        llm_user_input = prompt.get('user_input', None)
        edit_prompt = prompt.get('edit_prompt', None)
        message = None
        if llm_user_input is not None:
            qwen_prompt = prompt.get("instruction_qwen", None)
            think_content = prompt.get("think_content", None)
            if qwen_prompt is not None:
                message = build_message(llm_user_input, think_content, qwen_prompt, None, None, use_template=True, logger=logger)
            else:
                message = build_message(llm_user_input, think_content, qwen_prompt, llm_processor, accelerator.unwrap_model(model_llm), use_template=False, logger=logger)

        use_edit_processor = False
        # max_input_image_size = 1024 if input_images is not None and len(input_images) == 1 else 512
        max_input_image_size = 512

        if edit_prompt is not None:
            think_content = prompt.get("think_content", None)
            message = build_message_edit(edit_prompt, input_images, think_content, max_input_image_size)
            use_edit_processor = True

        images = pipe(
            prompt=prompt_,
            llm_user_input=llm_user_input,
            pure_text=message,
            height=val_img_size,
            width=val_img_size,
            guidance_scale=2.5 if input_images is not None else 3,
            img_guidance_scale=1.6 if input_images is not None else None,
            # guidance_scale=1.0,
            # img_guidance_scale=1.0,
            seed=42,
            input_images=input_images,
            input_llm_images=input_images,
            use_input_image_size_as_output=True if input_images is not None and len(input_images) == 1 else False,
            use_edit_processor=use_edit_processor,
            max_input_image_size=max_input_image_size,
        )
        images_result.append(images[0])

        if input_images is not None and len(input_images) > 0:
            # input_image_list.append([crop_arr(Image.open(item), max_image_size=max_input_image_size) for item in input_images])
            input_image_list.append([crop_by_max_pixels(Image.open(item), max_input_image_size * max_input_image_size) for item in input_images])
        else:
            input_image_list.append([])
        output_image = prompt.get("output_image", None)
        if output_image is not None:
            # output_image_gt_result.append(crop_arr(Image.open(output_image), max_image_size=max_input_image_size))
            output_image_gt_result.append(crop_by_max_pixels(Image.open(output_image), max_input_image_size * max_input_image_size))

    save_path = os.path.join(base_dir, f'{int(step):07d}.jpg')

    # concatenated_image = Image.new('RGB', (images_result[0].width * len(images_result), images_result[0].height))
    # for idx, img in enumerate(images_result):
    #     loc = idx * images_result[0].width
    #     concatenated_image.paste(img, (loc, 0))  # img1 放在左边

    if len(output_image_gt_result) == 0:
        width_list = []
        height_list = []
        max_input_image_length = [len(item) for item in input_image_list]
        for idx, img in enumerate(images_result):
            img_width, img_height = img.width, img.height
            if len(input_image_list[idx]) > 0:
                input_width_max = max([item.width for item in input_image_list[idx]])
                input_height_max = max([item.height for item in input_image_list[idx]])
                width_list.append(max(img_width, input_width_max))
                height_list.append(max(img_height, input_height_max))
            else:
                width_list.append(img_width)
                height_list.append(img_height)
        total_width = max(width_list) * (max(max_input_image_length) + 1)
        total_height = sum(height_list)
        concatenated_image = Image.new('RGB', (total_width, total_height))
        for idx, img in enumerate(images_result):
            if idx > 0:
                y = sum(height_list[:idx])
            else:
                y = 0
            input_images_temp = input_image_list[idx]
            start_width = 0
            for item in input_images_temp:
                concatenated_image.paste(item, (start_width, y))
                start_width += item.width
            concatenated_image.paste(img, (start_width, y))
        try:
            concatenated_image.save(save_path)
        except Exception as e:
            print(e)
    else:
        width_list = []
        height_list = []
        max_input_image_length = [len(item) for item in input_image_list]
        for idx, img in enumerate(images_result):
            img_width, img_height = img.width, img.height
            if len(input_image_list[idx]) > 0:
                input_width_max = max([item.width for item in input_image_list[idx]])
                input_height_max = max([item.height for item in input_image_list[idx]])
                gt_width, gt_height = output_image_gt_result[idx].size
                width_list.append(max(img_width, input_width_max, gt_width))
                height_list.append(max(img_height, input_height_max, gt_height))
            else:
                gt_width, gt_height = output_image_gt_result[idx].size
                width_list.append(max(img_width, gt_width))
                height_list.append(max(img_height, gt_height))
        total_width = max(width_list) * (max(max_input_image_length) + 2)
        total_height = sum(height_list)
        concatenated_image = Image.new('RGB', (total_width, total_height))
        for idx, img in enumerate(images_result):
            if idx > 0:
                y = sum(height_list[:idx])
            else:
                y = 0
            input_images_temp = input_image_list[idx]
            start_width = 0
            for item in input_images_temp:
                concatenated_image.paste(item, (start_width, y))
                start_width += item.width
            concatenated_image.paste(img, (start_width, y))
            start_width += img.width
            concatenated_image.paste(output_image_gt_result[idx], (start_width, y))
        try:
            concatenated_image.save(save_path)
        except Exception as e:
            print(e)

    model.train()
    if model_llm_training_flag:
        model_llm.train()
    return
