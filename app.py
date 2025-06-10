import os
import argparse
from functools import partial

import torch
import random
import spaces
import gradio as gr
from src import MindOmni

NEGATIVE_PROMPT = '''
low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.
'''


def parse_args():
    args = argparse.ArgumentParser(description='MindOmni')
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('--dtype', type=str, default='bf16')
    args.add_argument('--server_name', type=str, default='127.0.0.1')
    args.add_argument('--port', type=int, default=8080)
    args.add_argument('--model_path', type=str,
                      default='EasonXiao-888/MindOmni')
    args = args.parse_args()
    return args


def build_model(args):
    device = args.device
    MindOmni_model = MindOmni.from_pretrained(args.model_path)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    MindOmni_model.to(device=device, dtype=dtype)
    MindOmni_model.eval()
    return MindOmni_model


@spaces.GPU(duration=180)
def understand_func(
        MindOmni_model, text, do_sample, temperature,
        max_new_tokens, input_llm_images):
    if input_llm_images is not None and not isinstance(input_llm_images, list):
        input_llm_images = [input_llm_images]
    answer = MindOmni_model.generate_text(
        text, input_llm_images, do_sample, temperature,
        max_new_tokens, only_understand=True)
    return answer


@spaces.GPU(duration=180)
def generate_func(
        MindOmni_model, text, use_cot, height, width, guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model, max_input_image_size, randomize_seed, save_images, do_sample, temperature, max_new_tokens, input_llm_images, only_understand):
    if input_llm_images is not None and not isinstance(input_llm_images, list):
        input_llm_images = [input_llm_images]

    if randomize_seed:
        seed = random.randint(0, 10000000)

    os.makedirs(os.path.dirname('/tmp/.unhold'), exist_ok=True)
    with open('/tmp/.unhold', 'w') as f:
        f.write('')
    output, prompt_ = MindOmni_model.generate_image(
        height, width, guidance_scale, inference_steps, separate_cfg_infer, offload_model, seed, max_input_image_size,
        text, NEGATIVE_PROMPT, input_llm_images, do_sample, temperature, max_new_tokens, only_understand, use_cot=use_cot)
    os.remove('/tmp/.unhold')

    img = output[0]

    if save_images:
        # Save All Generated Images
        from datetime import datetime
        # Create outputs directory if it doesn't exist
        os.makedirs('assets/outputs', exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('assets/outputs', f'{timestamp}.png')
        # Save the image
        img.save(output_path)

    return img, prompt_, seed


def build_gradio(args, MindOmni_model):
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
                            g_use_cot = gr.Checkbox(label="With thinking", value=False)
                            g_do_sample = gr.Checkbox(label="Do sample", value=False)
                            g_temperature = gr.Slider(0, 10, value=1, label="Temperature")
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
                            g_offload = gr.Checkbox(label="Offload model to CPU", value=False)
                            g_save = gr.Checkbox(label="Save generated images", value=False)

                    with gr.Column(scale=1):
                        g_out_img = gr.Image(label="Generated Image")
                        g_prompt_out = gr.Textbox(label="MindOmni CoT Content")
                        g_seed_out = gr.Textbox(label="Used seed")

                        with gr.Accordion("🖼️ Prompt Examples: Text-only"):
                            gr.Examples(
                                examples=[
                                    ["Futuristic city skyline at sunset, digital art", 42, False, False, False, 1024, 1024, "assets/example_outputs/case_1.png"],
                                    ["An image of multiple apples, the quantity of apples is the solution of '2x + 6 = 16'.", 1723284, False, True, False, 512, 1024, "assets/example_outputs/case_2.png"],
                                    ["A park with benches equal to the solution of 'x^2 -2x = 8'.", 4318852, False, True, False, 512, 512, "assets/example_outputs/case_3.png"],
                                    ["An image of China's national treasure animal.", 42, False, True, False, 1024, 1024, "assets/example_outputs/case_4.png"],
                                    ["Scene in the Sydney Opera House when New York is at noon.", 42, False, True, False, 1024, 1024, "assets/example_outputs/case_5.png"],
                                    ["Generate an image of an animal with (3 + 6) lives", 7393438, False, True, False, 1024, 1024, "assets/example_outputs/case_6.png"],
                                ],
                                inputs=[g_prompt, g_seed, g_rand, g_use_cot, g_do_sample, g_height, g_width, g_out_img],
                            )
                        with gr.Accordion("🖼️ Prompt Examples: With reference image"):
                            gr.Examples(
                                examples=[
                                    ["An image of the animal growing up", "assets/tapdole.jpeg", 42, False, True, True, 1024, 1024, "assets/example_outputs/case_7.png"]
                                ],
                                inputs=[g_prompt, g_image, g_seed, g_rand, g_use_cot, g_do_sample, g_height, g_width, g_out_img],
                            )

                g_btn.click(
                    partial(generate_func, MindOmni_model),
                    inputs=[g_prompt, g_use_cot, g_height, g_width, g_scale, g_steps,
                            g_seed, g_sep_cfg, g_offload, g_max_img, g_rand, g_save,
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
                    partial(understand_func, MindOmni_model),
                    inputs=[u_prompt, u_do_sample,
                            u_temperature, u_max_new_tok, u_image],
                    outputs=u_answer)

        demo.launch(server_name=args.server_name, server_port=args.port)


def main():
    args = parse_args()
    print(f'running args: {args}')
    MindOmni_model = build_model(args)
    build_gradio(args, MindOmni_model)


if __name__ == '__main__':
    main()
