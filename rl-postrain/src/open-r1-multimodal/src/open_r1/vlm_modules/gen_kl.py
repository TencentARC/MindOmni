from OmniGen.utils import vae_encode, center_crop_arr
from OmniGen.train_helper import training_losses
import torch
from torchvision import transforms
import random
import torch.nn.functional as F


crop_func = center_crop_arr
image_transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: crop_func(pil_image, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

# def compute_gen_kl_loss(output_images, gt_prompt, messages, vae, processor, llm_processor, model_llm, model, weight_dtype, output_hidden_states_, attention_mask, input_ids, ref_hidden_states):
#     import ipdb; ipdb.set_trace()
#     loss_kl_list = []
#     model.llm.gradient_checkpointing_enable()
#     for idx, (output_image, prompt, message) in enumerate(zip(output_images, gt_prompt, messages)):
#         device = model_llm.device
#         output_image_ = image_transform(output_image).unsqueeze(0).to(device)
#         _, _, h, w = output_image_.shape

#         output_image_ = vae_encode(vae, output_image_, weight_dtype)
#         input_data = processor(instructions=[prompt], height=h, width=w, use_img_cfg=False, separate_cfg_input=True)
#         input_data = {key: value[0] for key, value in input_data.items()}   # use 0-index for data, not 1-index for cfg

#         model_kwargs = dict(input_ids=input_data['input_ids'].to(device), input_img_latents=None, input_image_sizes=input_data['input_image_sizes'], attention_mask=input_data['attention_mask'].to(device), position_ids=input_data['position_ids'].to(device), padding_latent=input_data['padding_images'], past_key_values=None, return_past_key_values=False)
#         # obtain the qwen feature
#         # import ipdb; ipdb.set_trace()

#         laten_noise_token_num = input_data['position_ids'].shape[1] - input_data['input_ids'].shape[1] - 1
#         valid_token_length = attention_mask[idx].sum()
#         output_hidden_states = output_hidden_states_[idx, :valid_token_length].unsqueeze(0)

#         llm_attention_mask = torch.tril(torch.ones(valid_token_length + 1 + laten_noise_token_num, valid_token_length + 1 + laten_noise_token_num))
#         llm_attention_mask[-laten_noise_token_num:, :] = torch.ones(laten_noise_token_num, valid_token_length + 1 + laten_noise_token_num)
#         llm_attention_mask = llm_attention_mask.unsqueeze(0).to(torch.int64).to(device)

#         connector_position_ids = torch.arange(valid_token_length, dtype=torch.int64).unsqueeze(0).to(device)
#         llm_position_ids = torch.arange(valid_token_length + 1 + laten_noise_token_num, dtype=torch.int64).unsqueeze(0).to(device)

#         func = model
#         hidden_states = func.qwen2phi[0](output_hidden_states)
#         cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
#         try:   # llm lora case
#             mask_func = model_llm.base_model.model.model._update_causal_mask
#         except Exception as e:  # noqa
#             mask_func = model_llm.model._update_causal_mask
#         cond_causal_mask = mask_func(
#             attention_mask[idx].unsqueeze(0), hidden_states, cache_position, None, None)
#         for decoder_layer in func.qwen2phi[1:]:
#             layer_out = decoder_layer(
#                 hidden_states,
#                 attention_mask=cond_causal_mask,
#                 position_ids=connector_position_ids,
#             )
#             hidden_states = layer_out[0]

#         # import ipdb; ipdb.set_trace()
#         model_kwargs['llm_input_embeds'] = hidden_states
#         model_kwargs['llm_attention_mask'] = llm_attention_mask
#         model_kwargs['llm_position_ids'] = llm_position_ids
#         model_kwargs['llm_image_sizes'] = {}

#         loss_kl = training_losses(model, output_image_, model_kwargs, use_dist=True, patch_weight=None)['dist_loss']
#         loss_kl_list.append(loss_kl)

#     return loss_kl_list


def compute_gen_kl_loss(output_images, gt_prompt, messages, vae, processor, llm_processor, model_llm, model, weight_dtype, output_hidden_states_, attention_mask, input_ids, ref_hidden_states):
    loss_kl_list = []
    model.train()
    model.llm.gradient_checkpointing_enable()
    h, w = output_images[0].size
    latent_size_h = h // 8
    latent_size_w = w // 8
    for idx, (hidden_states_, prompt, ref_hidden_states_) in enumerate(zip(output_hidden_states_, gt_prompt, ref_hidden_states)):
        device = model_llm.device

        latents = torch.randn(1, 4, latent_size_h, latent_size_w, device=model_llm.device, dtype=weight_dtype)
        t = torch.linspace(0, 1, 50 + 1)
        t = t / (t + 1 - 1 * t)
        t_idx = random.randint(0, 50)
        sigma_i = t[t_idx]
        timesteps = torch.zeros(size=(len(latents), )).to(latents.device) + sigma_i

        input_data = processor(instructions=[prompt], height=h, width=w, use_img_cfg=False, separate_cfg_input=True)
        input_data = {key: value[0] for key, value in input_data.items()}   # use 0-index for data, not 1-index for cfg

        model_kwargs = dict(input_ids=input_data['input_ids'].to(device), input_img_latents=None, input_image_sizes=input_data['input_image_sizes'], attention_mask=input_data['attention_mask'].to(device), position_ids=input_data['position_ids'].to(device), padding_latent=input_data['padding_images'], past_key_values=None, return_past_key_values=False)
        # obtain the qwen feature
        # import ipdb; ipdb.set_trace()

        laten_noise_token_num = input_data['position_ids'].shape[1] - input_data['input_ids'].shape[1] - 1
        valid_token_length = attention_mask[idx].sum()
        output_hidden_states = hidden_states_[:valid_token_length].unsqueeze(0)
        ref_output_hidden_states = ref_hidden_states_[:valid_token_length].unsqueeze(0)

        llm_attention_mask = torch.tril(torch.ones(valid_token_length + 1 + laten_noise_token_num, valid_token_length + 1 + laten_noise_token_num))
        llm_attention_mask[-laten_noise_token_num:, :] = torch.ones(laten_noise_token_num, valid_token_length + 1 + laten_noise_token_num)
        llm_attention_mask = llm_attention_mask.unsqueeze(0).to(torch.int64).to(device)

        connector_position_ids = torch.arange(valid_token_length, dtype=torch.int64).unsqueeze(0).to(device)
        llm_position_ids = torch.arange(valid_token_length + 1 + laten_noise_token_num, dtype=torch.int64).unsqueeze(0).to(device)

        func = model

        def forward_func(output_hidden_states):
            hidden_states = func.qwen2phi[0](output_hidden_states)
            cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
            try:   # llm lora case
                mask_func = model_llm.base_model.model.model._update_causal_mask
            except Exception as e:  # noqa
                mask_func = model_llm.model._update_causal_mask
            cond_causal_mask = mask_func(
                attention_mask[idx].unsqueeze(0), hidden_states, cache_position, None, None)
            for decoder_layer in func.qwen2phi[1:]:
                layer_out = decoder_layer(
                    hidden_states,
                    attention_mask=cond_causal_mask,
                    position_ids=connector_position_ids,
                )
                hidden_states = layer_out[0]

            model_kwargs['llm_input_embeds'] = hidden_states
            model_kwargs['llm_attention_mask'] = llm_attention_mask
            model_kwargs['llm_position_ids'] = llm_position_ids
            model_kwargs['llm_image_sizes'] = {}

            model_output, all_hidden_states = model(latents, timesteps, **model_kwargs, use_dist=True)
            return model_output, all_hidden_states
        model_output, all_hidden_states = forward_func(output_hidden_states)
        with torch.no_grad():
            ref_model_output, ref_all_hidden_states = forward_func(ref_output_hidden_states)

        temperature0 = 3
        dist_loss = []
        for i in range(len(all_hidden_states)):
            batch_dist_loss = F.kl_div(F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1).log(), F.softmax(normalize(ref_all_hidden_states[i])/temperature0, dim=-1), reduction='batchmean')
            dist_loss.append(batch_dist_loss)
        loss_kl = torch.stack(dist_loss).mean()

        # import ipdb; ipdb.set_trace()
        # print(1)
        loss_kl_list.append(loss_kl)

    model.eval()
    return loss_kl_list
