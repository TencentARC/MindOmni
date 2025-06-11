import spaces
import os
import argparse
from functools import partial

import torch
import random
import gradio as gr
from src import MindOmni

NEGATIVE_PROMPT = '''
low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.
'''

MindOmni_model = MindOmni.from_pretrained('EasonXiao-888/MindOmni')
MindOmni_model.to(device='cuda', dtype=torch.bfloat16)
MindOmni_model.eval()


@spaces.GPU
def understand_func(
        text, do_sample, temperature,
        max_new_tokens, input_llm_images):
    if input_llm_images is not None and not isinstance(input_llm_images, list):
        input_llm_images = [input_llm_images]
        
    answer = MindOmni_model.generate_text(
        text, input_llm_images, do_sample, temperature,
        max_new_tokens, only_understand=True)
    return answer


@spaces.GPU
def generate_func(
        text, use_cot, cascade_thinking, height, width, guidance_scale, inference_steps, seed, separate_cfg_infer, max_input_image_size, randomize_seed, do_sample, temperature, max_new_tokens, input_llm_images, only_understand):
    if input_llm_images is not None and not isinstance(input_llm_images, list):
        input_llm_images = [input_llm_images]
        
    if randomize_seed:
        seed = random.randint(0, 10000000)

    print(f'Generate image prompt: {text}')
    offload_model = False
    output, prompt_ = MindOmni_model.generate_image(
        height, width, guidance_scale, inference_steps, separate_cfg_infer, offload_model, seed, max_input_image_size,
        text, NEGATIVE_PROMPT, input_llm_images, do_sample, temperature, max_new_tokens, only_understand, use_cot=use_cot,
        cascade_thinking=cascade_thinking)
    print('Generation finished.')

    img = output[0]
    return img, prompt_, seed


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🪄 MindOmni Demo")

        with gr.Tabs():
            # ---------- GENERATE ----------
            with gr.TabItem("🎨 Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        g_prompt = gr.Textbox(label="Text prompt")
                        g_image = gr.Image(label="Condition image (optional)", type="filepath")
                        g_btn = gr.Button("🚀 Generate Image")

                        with gr.Accordion("📚 Image Generation Args"):
                            g_use_cot = gr.Checkbox(label="Use thinking", value=True)
                            g_cascade_thinking = gr.Checkbox(label="Cascade thinking (experimental for better quality)", value=False)
                            g_do_sample = gr.Checkbox(label="Do sample (for more diversity)", value=False)
                            g_temperature = gr.Slider(0, 10, value=0.6, label="Temperature")
                            g_max_new_tok = gr.Slider(32, 8192, value=512, label="Max new tokens")

                            g_height = gr.Slider(128, 2048, value=1024, step=16, label="Height")
                            g_width = gr.Slider(128, 2048, value=1024, step=16, label="Width")
                            g_scale = gr.Slider(1.0, 5.0, value=3.0, step=0.1, label="Guidance Scale")
                            g_steps = gr.Slider(1, 100, value=50, label="Inference Steps")
                            g_seed = gr.Slider(0, 2**31 - 1, value=42, label="Seed")
                            g_rand = gr.Checkbox(label="Randomize seed", value=False)
                            g_max_img = gr.Slider(128, 2048, value=1024, step=16,
                                                  label="Max input image size")
                            g_sep_cfg = gr.Checkbox(label="Separate-CFG infer", value=True)

                    with gr.Column(scale=1):
                        g_out_img = gr.Image(label="Generated Image")
                        g_prompt_out = gr.Textbox(label="MindOmni CoT Content")
                        g_seed_out = gr.Textbox(label="Used seed")

                        with gr.Accordion("🖼️ Prompt Examples: Text-only"):
                            gr.Examples(
                                examples=[
                                    ["Futuristic city skyline at sunset, digital art", None, 42, False, False, False, False, 1024, 1024, "assets/example_outputs/case_1.png"],
                                    ["An image of China's national treasure animal.", None, 42, False, True, False, False, 1024, 1024, "assets/example_outputs/case_2.png"],
                                    ["Scene in the Sydney Opera House when New York is at noon.", None, 42, False, True, False, False, 1024, 1024, "assets/example_outputs/case_3.png"],
                                    ["Generate an image of an animal with (3 + 6) lives", None, 7393438, False, True, False, False, 1024, 1024, "assets/example_outputs/case_4.png"],
                                ],
                                inputs=[g_prompt, g_image, g_seed, g_rand, g_use_cot, g_cascade_thinking, g_do_sample, g_height, g_width, g_out_img],
                            )
                        with gr.Accordion("🖼️ Prompt Examples: With reference image"):
                            gr.Examples(
                                examples=[
                                    ["An image of the animal growing up", "assets/tapdole.jpeg", 42, False, True, False, True, 1024, 1024, "assets/example_outputs/case_5.png"],
                                    ["Show a girl holding this plant in the autumn.", "assets/leaf.jpg", 42, False, True, True, False, 1024, 1024, "assets/example_outputs/case_6.png"]
                                ],
                                inputs=[g_prompt, g_image, g_seed, g_rand, g_use_cot, g_cascade_thinking, g_do_sample, g_height, g_width, g_out_img],
                            )

                g_btn.click(
                    generate_func,
                    inputs=[g_prompt, g_use_cot, g_cascade_thinking, g_height, g_width, g_scale,
                            g_steps, g_seed, g_sep_cfg, g_max_img, g_rand,
                            g_do_sample, g_temperature, g_max_new_tok,
                            g_image, gr.State(False)],          # only_understand=False
                    outputs=[g_out_img, g_prompt_out, g_seed_out])

            # ---------- UNDERSTAND ----------
            with gr.TabItem("🧠 Understand"):
                with gr.Row():
                    with gr.Column(scale=1):
                        u_prompt = gr.Textbox(label="Text prompt")
                        u_image = gr.Image(label="Image (optional)", type="filepath")
                        u_btn = gr.Button("🔍 Understand")
                        with gr.Accordion("📚 Text Generation Args"):
                            u_do_sample = gr.Checkbox(label="Do sample", value=False)
                            u_temperature = gr.Slider(0, 10, value=1, label="Temperature")
                            u_max_new_tok = gr.Slider(32, 8192, value=512, label="Max new tokens")

                    with gr.Column(scale=1):
                        u_answer = gr.Textbox(label="Answer", lines=8)

                u_btn.click(
                    understand_func,
                    inputs=[u_prompt, u_do_sample,
                            u_temperature, u_max_new_tok, u_image],
                    outputs=u_answer)
                
            # ---------- MULTIPLE IMAGES EDITING (Coming Soon) ----------
            with gr.TabItem("🖼️ Fine-grained Editing"):
                with gr.Column():
                    gr.Markdown("🚧 **Coming Soon**: Support for fine-grained editing on single/multiple images will be available soon.")

        demo.launch()


if __name__ == '__main__':
    build_gradio()