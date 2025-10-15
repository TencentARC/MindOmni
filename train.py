import json
from time import time
import argparse
import logging
import os
from pathlib import Path
import math

import numpy as np
from PIL import Image
from copy import deepcopy
import shutil

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torch_npu

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen import forward as qwen_forward
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator, X2IWebDataset, ShortLongWebDataset
from OmniGen.train_helper import training_losses, validate_func
from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list
)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import transformers
# get the hidden_states[-1] before the normalization
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen_forward


def main(args):
    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        kwargs_handlers=[kwargs],
        )
    device = accelerator.device
    t2i_keys = getattr(args, 'keys')
    delattr(args, 'keys')   # accelerate init_trackers not allow list args
    accelerator.init_trackers("tensorboard_log", config=args.__dict__)

    # set seed
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    logger = create_logger(args.results_dir)
    checkpoint_dir = f"{args.results_dir}/checkpoints"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, 'train_args.json'), 'w'), indent=4)


    # Create model:    
    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.model_name_or_path = snapshot_download(repo_id=args.model_name_or_path,
                                        cache_dir=cache_folder,
                                        ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        logger.info(f"Downloaded model to {args.model_name_or_path}")
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()

    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            logger.info("If you have VAE in local folder, please specify the path with --vae_path")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=torch.float32)
    model.to(weight_dtype)

    logger.info("Preparing llm model")
    model_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_llm)
    model_llm.to(weight_dtype)
    requires_grad(model_llm, False)
    model_llm.eval()
    if args.mllm_ckpt is not None:    # load original mllm or grpo ckpt
        # model_llm = PeftModel.from_pretrained(model_llm, args.mllm_ckpt)
        grpo_mllm_adapter = '/group/40034/yichengxiao/VLM-R1/checkpoints/rl/Qwen2.5-VL-7B-Instruct-ust_omnigen-lora_exp4/checkpoint-300'
        model_llm = PeftModel.from_pretrained(model_llm, grpo_mllm_adapter)
        model_llm = model_llm.merge_and_unload()
        # logger.info(f"Pretrained MLLM load from {args.mllm_ckpt}")
        logger.info(f"Pretrained MLLM load from {grpo_mllm_adapter}")
    llm_processor = AutoProcessor.from_pretrained(args.model_llm)
    special_tokens_dict = {
        "additional_special_tokens": ["<img>", "</img>"],
    }
    num_new_tokens = llm_processor.tokenizer.add_special_tokens(special_tokens_dict)
    model_llm.resize_token_embeddings(len(llm_processor.tokenizer))
    tokenizer_dir = f"{args.results_dir}/llm_tokenizer"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        os.makedirs(tokenizer_dir, exist_ok=True)
        llm_processor.tokenizer.save_pretrained(tokenizer_dir)
        input_embeddings = model_llm.get_input_embeddings()
        torch.save(input_embeddings.state_dict(), f"{tokenizer_dir}/input_embedding.pt")

    if args.mllm_ckpt is not None:   # load only train mllm ckpt
        model_llm = PeftModel.from_pretrained(model_llm, args.mllm_ckpt)
        model_llm = model_llm.merge_and_unload()
        logger.info(f"Pretrained MLLM load from {args.mllm_ckpt}")

    if args.use_llm_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model_llm, False)
        transformer_lora_config = LoraConfig(
            # r=args.lora_rank,
            # lora_alpha=args.lora_rank,
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model_llm.enable_input_require_grads()
        model_llm = get_peft_model(model_llm, transformer_lora_config)
        model_llm.to(weight_dtype)
        model_llm_lora_parameters = list(filter(lambda p: p.requires_grad, model_llm.parameters()))

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    requires_grad(vae, False)
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            # target_modules=["qkv_proj", "o_proj"],
            target_modules=r"llm\.layers\.\d+\.self_attn\.(qkv_proj|o_proj)",
        )
        model.llm.enable_input_require_grads()
        if args.pretrained_model is not None:
            base_pretrained_path = '/'.join(args.pretrained_model.split('/')[:-1])
            peft_pretrained_json = os.path.join(base_pretrained_path, 'adapter_config.json')
            if os.path.exists(peft_pretrained_json):
                model = PeftModel.from_pretrained(model.cpu(), base_pretrained_path).to(device)
            else:
                model = get_peft_model(model, transformer_lora_config)
        else:
            model = get_peft_model(model, transformer_lora_config)
        model.to(weight_dtype)

    if args.pretrained_model is not None:
        qwen2phi_module_origin = torch.load(args.pretrained_model, map_location='cpu')
        if args.only_load_decoder:
            del qwen2phi_module_origin['0.weight']
            del qwen2phi_module_origin['0.bias']
            logger.info('only load decoder part of connector')

        module_keys = list(model.qwen2phi.state_dict().keys())
        pretrained_keys = list(qwen2phi_module_origin.keys())
        all_keys = module_keys + pretrained_keys
        missing_modules = []
        unexpected_modules = []
        for item in all_keys:
            if item in module_keys and item not in qwen2phi_module_origin.keys():
                missing_modules.append(item)
            if item not in module_keys and item in qwen2phi_module_origin.keys():
                unexpected_modules.append(item)

        logger.info(f"loading {model.qwen2phi.__class__.__name__} but missing modules: {missing_modules}, unexpected modules: {unexpected_modules}")
        model.qwen2phi.load_state_dict(qwen2phi_module_origin, strict=False)
        logger.info(f"Successfully loading {args.pretrained_model}")

    if args.freeze_omnigen:
        requires_grad(model, False)
    # train additional modules
    additional_train_modules = {'qwen2phi', 'lora'}
    for name, module in model.named_modules():
        for train_name in additional_train_modules:
            if train_name in name:
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.use_llm_lora:
        transformer_lora_parameters = transformer_lora_parameters + model_llm_lora_parameters
    opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)


    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    

    # Setup data:
    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    max_image_size_webdataset = args.max_image_size_webdataset if args.max_image_size_webdataset is not None else args.max_image_size
    image_transform_webdataset = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, max_image_size_webdataset)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if not args.only_t2i:
        dataset = DatasetFromJson(
            json_file=args.json_file,
            image_path=args.image_path,
            processer=processor,
            image_transform=image_transform,
            image_transform_webdataset=image_transform_webdataset,
            max_input_length_limit=args.max_input_length_limit,
            condition_dropout_prob=args.condition_dropout_prob,
            keep_raw_resolution=args.keep_raw_resolution,
            x2i_length=args.x2i_length,
            t2i_length=args.t2i_length,
            t2i_ratio=args.t2i_ratio,
            t2i_json=args.t2i_json,
            t2i_keys=t2i_keys,
            t2i_file_name=args.file_name,
            use_longshort_t2i=args.use_longshort_t2i,
            llm_processor=llm_processor,
            llm_type="qwen2.5vl",
            sub_json=args.sub_json,
            sub_file_name=args.sub_file_name,
        )
    else:
        # dataset = X2IWebDataset(
        #     data_root=args.t2i_json,
        #     file_name=args.file_name,
        #     keys=t2i_keys,
        #     processer=processor,
        #     image_transform=image_transform_webdataset,
        #     max_input_length_limit=args.max_input_length_limit,
        #     condition_dropout_prob=args.condition_dropout_prob,
        #     keep_raw_resolution=args.keep_raw_resolution,
        #     llm_processor=llm_processor,
        #     llm_type="qwen2.5vl",
        # )
        dataset = ShortLongWebDataset(
            data_root=args.t2i_json,
            file_name=args.file_name,
            keys=t2i_keys,
            processer=processor,
            image_transform=image_transform_webdataset,
            max_input_length_limit=args.max_input_length_limit,
            condition_dropout_prob=args.condition_dropout_prob,
            keep_raw_resolution=args.keep_raw_resolution,
            llm_processor=llm_processor,
            llm_type="qwen2.5vl",
        )
    collate_fn = TrainDataCollator(pad_token_id=processor.text_tokenizer.eos_token_id, llm_pad_token_id=llm_processor.tokenizer.eos_token_id,hidden_size=model.llm.config.hidden_size, keep_raw_resolution=args.keep_raw_resolution)

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,}")

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    if ema is not None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode

    if args.resume:
        emb_path = os.path.join('/'.join(args.pretrained_model.split('/')[:2]), 'llm_tokenizer', 'input_embedding.pt')
        model_llm.model.embed_tokens.load_state_dict(torch.load(emb_path, map_location='cpu'))
        opt_path = os.path.join('/'.join(args.pretrained_model.split('/')[:-1]), 'opt.pt')
        opt.load_state_dict(torch.load(opt_path, map_location='cpu'))
        start_step = int(args.pretrained_model.split('/')[-2])
        if accelerator.is_main_process:
            logger.info(f'reusme training from step {start_step}')

    # if ema is not None:
    #     model, ema = accelerator.prepare(model, ema)
    # else:
    #     model = accelerator.prepare(model)

    # opt, loader, lr_scheduler = accelerator.prepare(opt, loader, lr_scheduler)
    assert ema is None, 'Not test the code with ema yet'
    model, model_llm, opt, loader, lr_scheduler = accelerator.prepare(model, model_llm, opt, loader, lr_scheduler)
    model_llm = model_llm.to(device)
    model = model.to(device)

    # Variables for monitoring/logging purposes:
    train_steps, log_steps = 0, 0
    if args.resume:
        train_steps = start_step
    running_loss = 0
    if args.use_dist:
        running_loss_dist = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
        # logger.info(f"Generate before the training")
        # validate_func(accelerator, model, vae, processor, model_llm, llm_processor, logger, base_dir=args.results_dir, step=0, device=device, val_img_size=args.val_img_size)

    accelerator.wait_for_everyone()
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        for data in loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    output_images = data['output_images']
                    input_pixel_values = data['input_pixel_values']
                    if isinstance(output_images, list):
                        output_images = vae_encode_list(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
                    else:
                        output_images = vae_encode(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
                
                # TODO: weighted loss for image editting
                # patch_weight = []
                # for i in range(len(output_images)):
                #     temp_x = output_images[i]
                #     w = torch.ones_like(temp_x).detach()
                #     if temp_x is for editing task:
                #         # Find the input image corresponding to the output image. We store the index in need_edit_imgs
                #         input_x = input_pixel_values[need_edit_imgs[i]]
                #         diff = torch.abs(temp_x - input_x).detach() # no grandient for weight
                #         diff_mean = torch.mean(diff)
                #         if diff_mean < 0.001:
                #             # The difference between the input and output images is too small, so we suspect there might be an issue with this data. We discard the image by setting its weight to zero.
                #             w = w * 0
                #         elif diff_mean <= 0.8:
                #             weight = 1 / (diff_mean + 1e-6)
                #             weight = max(min(weight, 64), 5) #crop the weight
                #             w[diff>0.3] = weight  #assign the weight to the pixels which are different in input and output
                #         else:
                #             # The difference between the input and output images is significant enough, so there's no need to reinforce the loss.
                #             pass
                #     patch_weight.append(w)

                # patch_weight = []
                # for i in range(len(output_images)):
                #     temp_x = output_images[i]
                #     w = torch.ones_like(temp_x).detach()
                #     if len(input_pixel_values) != 0:    # <cfg>
                #         input_x = input_pixel_values[i]

                #         diff = torch.abs(temp_x - input_x).detach() # no grandient for weight
                #         diff_mean = torch.mean(diff)
                #         if diff_mean < 0.001:
                #             # The difference between the input and output images is too small, so we suspect there might be an issue with this data. We discard the image by setting its weight to zero.
                #             w = w * 0
                #         elif diff_mean <= 0.8:
                #             # import ipdb; ipdb.set_trace()
                #             weight = 1 / (diff_mean + 1e-6)
                #             weight = max(min(weight, 64), 5) #crop the weight
                #             w[diff > 0.3] = weight  #assign the weight to the pixels which are different in input and output
                #         else:
                #             # The difference between the input and output images is significant enough, so there's no need to reinforce the loss.
                #             pass
                #     patch_weight.append(w)
                
                # import ipdb; ipdb.set_trace()
                patch_weight = None

                model_kwargs = dict(input_ids=data['input_ids'], input_img_latents=input_pixel_values, input_image_sizes=data['input_image_sizes'], attention_mask=data['attention_mask'], position_ids=data['position_ids'], padding_latent=data['padding_images'], past_key_values=None, return_past_key_values=False)
                
                # obtain the qwen feature
                # llm_inputs = dict(input_ids=data['llm_padded_input_ids'], attention_mask=data['llm_vae_attention_mask'], output_hidden_states=True)
                llm_inputs = dict(input_ids=data['llm_padded_input_ids'],
                                  attention_mask=data['llm_vae_attention_mask'],
                                  position_ids=data['llm_vae_position_ids'],
                                  output_hidden_states=True)

                if args.use_llm_lora:
                    # ### Only T2I Mode:
                    # llm_outputs = model_llm(**llm_inputs)
                    # output_hidden_states = llm_outputs.hidden_states[-1]

                    # ### TODO: X2I Mode: only support batch_size = 1
                    for i in range(len(data['llm_padded_input_ids'])):
                        pixel_values = None
                        image_grid_thw = None
                        # import ipdb; ipdb.set_trace()
                        # TODO: only support batchsize=1 when input image to qwen
                        # assert len(input_data['llm_inputs']) == 1, 'only support batchsize=1 when input image to qwen'
                        if data['llm_inputs'][i] is not None:
                            pixel_values = torch.tensor(data['llm_inputs'][i]['pixel_values']).to(device)
                            image_grid_thw = torch.tensor(data['llm_inputs'][i]['image_grid_thw']).to(device)

                        llm_inputs = dict(**llm_inputs,
                                          pixel_values=pixel_values,
                                          image_grid_thw=image_grid_thw,
                                          )
                        llm_outputs = model_llm(**llm_inputs)
                        output_hidden_states = llm_outputs.hidden_states[-1]
                else:
                    with torch.no_grad():
                        # ### Only T2I Mode:
                        # llm_outputs = model_llm(**llm_inputs)
                        # output_hidden_states = llm_outputs.hidden_states[-1]

                        # ### TODO: X2I Mode: only support batch_size = 1
                        for i in range(len(data['llm_padded_input_ids'])):
                            pixel_values = None
                            image_grid_thw = None
                            # import ipdb; ipdb.set_trace()
                            # TODO: only support batchsize=1 when input image to qwen
                            # assert len(input_data['llm_inputs']) == 1, 'only support batchsize=1 when input image to qwen'
                            if data['llm_inputs'][i] is not None:
                                pixel_values = torch.tensor(data['llm_inputs'][i]['pixel_values']).to(device)
                                image_grid_thw = torch.tensor(data['llm_inputs'][i]['image_grid_thw']).to(device)

                            llm_inputs = dict(**llm_inputs,
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw,
                                              )
                            llm_outputs = model_llm(**llm_inputs)
                            # try:
                            #     llm_outputs = model_llm(**llm_inputs)
                            # except Exception as e:
                            #     import ipdb; ipdb.set_trace()
                            #     print(e)
                            output_hidden_states = llm_outputs.hidden_states[-1]

                if hasattr(model, "module"):
                    func = model.module
                else:
                    func = model
                hidden_states = func.qwen2phi[0](output_hidden_states)
                cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                if args.use_llm_lora:
                    cond_causal_mask = accelerator.unwrap_model(model_llm).base_model.model.model._update_causal_mask(
                        data['llm_vae_attention_mask'], hidden_states, cache_position, None, None)
                else:
                    cond_causal_mask = accelerator.unwrap_model(model_llm).model._update_causal_mask(
                        data['llm_vae_attention_mask'], hidden_states, cache_position, None, None)
                for decoder_layer in func.qwen2phi[1:]:
                    layer_out = decoder_layer(
                        hidden_states,
                        attention_mask=cond_causal_mask,
                        position_ids=data['llm_vae_position_ids'],
                    )
                    hidden_states = layer_out[0]

                # import ipdb; ipdb.set_trace()
                model_kwargs['llm_input_embeds'] = hidden_states
                model_kwargs['llm_attention_mask'] = data['llm_attention_mask']
                model_kwargs['llm_position_ids'] = data['llm_position_ids']
                model_kwargs['llm_padded_input_ids'] = data['llm_padded_input_ids']
                model_kwargs['llm_image_sizes'] = data['llm_image_sizes']

                loss_dict = training_losses(model, output_images, model_kwargs, use_dist=args.use_dist, patch_weight=patch_weight)
                if args.use_dist:
                    loss = loss_dict["loss"] + loss_dict["dist_loss"] * args.dist_coef
                else:
                    loss = loss_dict["loss"]

                if args.use_dist:
                    running_loss += loss_dict["loss"].item()
                    running_loss_dist += loss_dict["dist_loss"].item()
                else:
                    running_loss += loss.item()
                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

                log_steps += 1
                train_steps += 1

                tracker_dict = {
                    "loss_diff": loss_dict["loss"].item(),
                }
                if args.use_dist:
                    tracker_dict['loss_dist'] = loss_dict["dist_loss"].item()
                accelerator.log(tracker_dict, step=train_steps)

                if train_steps % args.gradient_accumulation_steps == 0:
                    if accelerator.sync_gradients and ema is not None:
                        update_ema(ema, model)
                    
                if train_steps % (args.log_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    else:
                        torch_npu.npu.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes 
                    if args.use_dist:
                        avg_loss_dist = torch.tensor(running_loss_dist / log_steps, device=device)
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(avg_loss_dist, op=dist.ReduceOp.SUM)
                        avg_loss_dist = avg_loss_dist.item() / accelerator.num_processes 
                       
                    if accelerator.is_main_process:
                        cur_lr = opt.param_groups[0]["lr"]
                        if args.use_dist:
                            logger.info(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Train Loss Dist: {avg_loss_dist:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")
                        else:
                            logger.info(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")

                    # Reset monitoring variables:
                    running_loss = 0
                    if args.use_dist:
                        running_loss_dist = 0
                    log_steps = 0
                    start_time = time()

            # import ipdb; ipdb.set_trace()
            # [name for name, param in model.named_parameters() if param.requires_grad]
            if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                if accelerator.distributed_type == DistributedType.FSDP:
                    state_dict = accelerator.get_state_dict(model)
                    ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None
                else:
                    if not args.use_lora:
                        if hasattr(model, "module"):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None

                if accelerator.is_main_process:
                    if args.use_lora:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        
                        if hasattr(model, "module"):
                            model.module.save_pretrained(checkpoint_path)
                        else:
                            model.save_pretrained(checkpoint_path)
                    else:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        if not args.freeze_omnigen:
                            torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
                            processor.text_tokenizer.save_pretrained(checkpoint_path)
                            if hasattr(model, "module"):
                                model.module.llm.config.save_pretrained(checkpoint_path)
                            else:
                                model.llm.config.save_pretrained(checkpoint_path)
                            if ema_state_dict is not None:
                                checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}_ema"
                                os.makedirs(checkpoint_path, exist_ok=True)
                                torch.save(ema_state_dict, os.path.join(checkpoint_path, "model.pt"))
                                processor.text_tokenizer.save_pretrained(checkpoint_path)
                                model.llm.config.save_pretrained(checkpoint_path)

                    # check and save the projector bridge qwen and omnigen
                    qwen2phi_state_dict = accelerator.unwrap_model(model).qwen2phi.state_dict()
                    opt_state_dict = opt.state_dict()
                    accelerator.save(qwen2phi_state_dict, os.path.join(checkpoint_path, "qwen2phi.pt"))
                    accelerator.save(opt_state_dict, os.path.join(checkpoint_path, "opt.pt"))
                    logger.info(f"Saved connectorcheckpoint to {checkpoint_path}")

                    if args.use_llm_lora:
                        llm_checkpoint_path = os.path.join(checkpoint_path, 'llm_lora')
                        os.makedirs(llm_checkpoint_path, exist_ok=True)
                        if hasattr(model, "module"):
                            model_llm.module.save_pretrained(llm_checkpoint_path)
                        else:
                            model_llm.save_pretrained(llm_checkpoint_path)

                    remove_step_id = int(train_steps / args.gradient_accumulation_steps) - args.num_save_ckpt * args.ckpt_every
                    if remove_step_id > 0:
                        remove_checkpoint_path = f"{checkpoint_dir}/{int(remove_step_id):07d}/"
                        if os.path.exists(remove_checkpoint_path):
                            shutil.rmtree(remove_checkpoint_path)

                    # validate model
                    # import ipdb; ipdb.set_trace()
                    validate_func(accelerator, model, vae, processor, model_llm, llm_processor, logger, base_dir=args.results_dir, step=train_steps, device=device, val_img_size=args.val_img_size)
                    
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
    accelerator.end_training()
    model.eval()  
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name_or_path", type=str, default="OmniGen")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--vae_path", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=1024)
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--x2i_length", type=int, default=None)
    parser.add_argument("--max_image_size_webdataset", type=int, default=None)
    parser.add_argument("--t2i_length", type=int, default=5000000)
    parser.add_argument("--use_longshort_t2i", action="store_true")
    parser.add_argument("--t2i_ratio", type=float, default=0.0)
    parser.add_argument("--model_llm", type=str, default="qwen2.5-VL")
    parser.add_argument("--sub_json", type=str, default=None)
    parser.add_argument("--sub_file_name", type=str, default=None)
    parser.add_argument("--t2i_json", type=str, default="/group/40121/public_datasets/X2I_DATA/X2I-text-to-image/laion-coco-aesthetic_webdataset")
    parser.add_argument("--file_name", type=str, default="tar.gz")
    parser.add_argument("--keys", nargs='+', default=['png', 'txt'])
    parser.add_argument("--val_img_size", type=int, default=512)
    parser.add_argument("--only_t2i", action="store_true")
    parser.add_argument("--freeze_omnigen", action="store_true")
    parser.add_argument("--only_load_decoder", action="store_true")
    parser.add_argument("--use_dist", action="store_true")
    parser.add_argument("--dist_coef", type=float, default=1.0)
    parser.add_argument("--diff_coef", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_llm_lora", action="store_true")
    parser.add_argument("--mllm_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--num_save_ckpt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--keep_raw_resolution",
        action="store_true",
        help="multiple_resolutions",
    )
    parser.add_argument("--max_image_size", type=int, default=1344)

    parser.add_argument(
            "--use_lora",
            action="store_true",
        )
    parser.add_argument(
            "--lora_rank",
            type=int, 
            default=8
        )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether or not to use ema.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    ) 
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )


    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    main(args)


