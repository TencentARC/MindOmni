import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger
from transformers.cache_utils import DynamicCache


class MindOmniMLLM_Model(Qwen2_5_VLModel):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer before the self.norm
        # import ipdb; ipdb.set_trace()
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MindOmniMLLM(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.model = MindOmniMLLM_Model(config)

    # @staticmethod
    # def _update_model_kwargs_for_generation(
    #     outputs, model_kwargs, past_key_values_field="past_key_values"
    # ):
    #     if past_key_values_field in outputs:
    #         model_kwargs[past_key_values_field] = outputs[past_key_values_field]

    #     if "attention_mask" in model_kwargs:
    #         bs, _ = model_kwargs["attention_mask"].shape
    #         new_mask = torch.ones(bs, 1, dtype=model_kwargs["attention_mask"].dtype,
    #                               device=model_kwargs["attention_mask"].device)
    #         model_kwargs["attention_mask"] = torch.cat(
    #             [model_kwargs["attention_mask"], new_mask], dim=-1
    #         )
    #     return model_kwargs

    # @staticmethod
    # def _sample_token(
    #     logits: torch.Tensor,
    #     do_sample: bool,
    #     logits_processors: LogitsProcessorList,
    #     temperature: float,
    #     top_p: float,
    # ):
    #     """do sample / greedy"""
    #     logits = logits_processors(None, logits)
    #     if do_sample:
    #         # 温度缩放
    #         if temperature != 1.0 and temperature > 0:
    #             logits = logits / temperature
    #         # nucleus
    #         if top_p < 1.0:
    #             logits = TopPLogitsWarper(top_p=top_p)(None, logits)
    #         probs = nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    #         next_token = torch.multinomial(probs, num_samples=1)
    #     else:  # greedy
    #         next_token = torch.argmax(logits, dim=-1, keepdim=True)
    #     return next_token

    # @torch.no_grad()
    # def generate(
    #     self,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     max_new_tokens: int = 64,
    #     do_sample: bool = False,
    #     temperature: float = 1.0,
    #     top_p: float = 0.95,
    #     device: Union[str, torch.device] = "cuda",
    # ) -> torch.LongTensor:

    #     assert input_ids is not None
    #     eos_token_id = self.config.eos_token_id

    #     generated = [input_ids]

    #     input_ids = input_ids.to(device)
    #     if pixel_values is not None:
    #         pixel_values = pixel_values.to(device)
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    #     logits_processors = LogitsProcessorList()
    #     if temperature != 1.0 and do_sample:
    #         logits_processors.append(TemperatureLogitsWarper(temperature))
    #     if top_p < 1.0 and do_sample:
    #         logits_processors.append(TopPLogitsWarper(top_p=top_p))

    #     # ---- 推理循环 ---- #
    #     model_kwargs = {
    #         "attention_mask": attention_mask,
    #         "use_cache": True,
    #         "past_key_values": None,
    #         "cache_position": torch.arange(attention_mask.shape[-1]).to(attention_mask)
    #     }

    #     for _ in range(max_new_tokens):
    #         model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    #         outputs = self(
    #             input_ids=input_ids,
    #             use_cache=True,
    #             **model_kwargs,
    #         )

    #         next_token = self._sample_token(
    #             outputs.logits[:, -1, :],
    #             do_sample=do_sample,
    #             logits_processors=logits_processors,
    #             temperature=temperature,
    #             top_p=top_p,
    #         )  # (bs, 1)

    #         # 追加生成
    #         input_ids = next_token
    #         generated.append(next_token)

    #         # 更新 kv cache / attention_mask
    #         model_kwargs = self._update_model_kwargs_for_generation(
    #             outputs, model_kwargs
    #         )

    #         # 判断终止：所有 batch 均生成 eos
    #         if eos_token_id is not None:
    #             if (next_token == eos_token_id).all():
    #                 break

    #     generated_ids = torch.cat(generated, dim=1)

    #     return generated_ids
