import os
from modal import App, Image, Volume, asgi_app, Secret, Mount

app = App("spatialgpt-llava3d")

hf_volume = Volume.from_name("llava3d-hf-cache", create_if_missing=True)
data_volume = Volume.from_name("llava3d-data", create_if_missing=True)

base_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "torch-scatter",
        find_links="https://data.pyg.org/whl/torch-2.1.0+cu118.html",
    )
    .pip_install(
        "transformers==4.37.2",
        "tokenizers==0.15.1",
        "sentencepiece==0.1.99",
        "shortuuid",
        "accelerate==0.21.0",
        "peft",
        "bitsandbytes",
        "pydantic",
        "markdown2[all]",
        "numpy",
        "scikit-learn==1.2.2",
        "gradio==4.29.0",
        "gradio_client==0.16.1",
        "requests",
        "httpx==0.27.0",
        "uvicorn",
        "fastapi",
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.6.13",
        "opencv-python",
        "protobuf",
        "gdown",
    )
)

GPU_TYPE = os.environ.get("SPATIALGPT_MODAL_GPU", "A10G")

@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    secrets=[Secret.from_name("huggingface")],
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/workspace/data": data_volume,
    },
    keep_warm=1,
    mounts=[
        Mount.from_local_dir(".", remote_path="/root/app")
    ],
)
@asgi_app()
def web():
    import os
    import re
    import time
    import sys
    import gradio as gr
    import torch
    from PIL import Image as PILImage
    from io import BytesIO
    import requests
    from fastapi import FastAPI

    sys.path.insert(0, "/root/app")

    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        process_images,
        process_videos,
        tokenizer_special_token,
        get_model_name_from_path,
    )
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.utils import disable_torch_init

    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.makedirs("/workspace/data/playground/data/annotations", exist_ok=True)
    cam_params_path = "/workspace/data/playground/data/annotations/camera_parameters.json"
    if not os.path.exists(cam_params_path):
        try:
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1a-1MCFLkfoXNgn9XdlmS9Gnzplrzw7vf",
                cam_params_path,
                quiet=True,
            )
        except Exception:
            pass

    model_id = os.environ.get("SPATIALGPT_MODEL", "ChaimZhu/LLaVA-3D-7B")
    precision = os.environ.get("SPATIALGPT_PRECISION", "bf16")
    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half

    tokenizer = None
    model = None
    processor = None
    context_len = None

    def ensure_model():
        nonlocal tokenizer, model, processor, context_len
        if model is None:
            disable_torch_init()
            name = get_model_name_from_path(model_id)
            t, m, p, c = load_pretrained_model(model_id, None, name, torch_dtype=torch_dtype)
            tokenizer = t
            model = m
            processor = p
            context_len = c

    def load_image(image_file):
        if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
            response = requests.get(image_file)
            image = PILImage.open(BytesIO(response.content)).convert("RGB")
        elif isinstance(image_file, str):
            image = PILImage.open(image_file).convert("RGB")
        else:
            image = PILImage.open(image_file).convert("RGB")
        return image

    def format_prompt(qs, model_name, model_ref):
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "3D" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        if IMAGE_PLACEHOLDER in qs:
            if model_ref.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model_ref.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def infer_image(image, prompt, temperature, top_p, num_beams, max_new_tokens):
        ensure_model()
        start = time.time()
        qs = prompt or "Describe the image."
        model_name = get_model_name_from_path(model_id)
        prompt_text = format_prompt(qs, model_name, model)
        images = [load_image(image)]
        images_tensor = process_images(
            images,
            processor["image"],
            model.config,
        ).to(model.device, dtype=torch_dtype)
        input_ids = tokenizer_special_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=None,
                poses=None,
                intrinsics=None,
                clicks=None,
                image_sizes=None,
                do_sample=True if temperature and temperature > 0 else False,
                temperature=temperature or 0.2,
                top_p=top_p,
                num_beams=num_beams or 1,
                max_new_tokens=max_new_tokens or 512,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start
        return outputs, f"{elapsed:.2f}s"

    def infer_scene(scene_path, prompt, temperature, top_p, num_beams, max_new_tokens):
        ensure_model()
        start = time.time()
        qs = prompt or "Describe the scene."
        model_name = get_model_name_from_path(model_id)
        prompt_text = format_prompt(qs, model_name, model)
        videos_dict = process_videos(
            scene_path,
            processor["video"],
            mode="random",
            device=model.device,
            text=qs,
        )
        images_tensor = videos_dict["images"].to(model.device, dtype=torch_dtype)
        depths_tensor = videos_dict["depths"].to(model.device, dtype=torch_dtype)
        poses_tensor = videos_dict["poses"].to(model.device, dtype=torch_dtype)
        intrinsics_tensor = videos_dict["intrinsics"].to(model.device, dtype=torch_dtype)
        clicks_tensor = torch.zeros((0, 3)).to(model.device, dtype=torch.bfloat16)
        input_ids = tokenizer_special_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                clicks=clicks_tensor,
                image_sizes=None,
                do_sample=True if temperature and temperature > 0 else False,
                temperature=temperature or 0.2,
                top_p=top_p,
                num_beams=num_beams or 1,
                max_new_tokens=max_new_tokens or 512,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - start
        return outputs, f"{elapsed:.2f}s"

    with gr.Blocks() as demo:
        gr.Markdown("LLaVA-3D on Modal")
        with gr.Tab("Image"):
            in_image = gr.Image(type="filepath", label="Image")
            in_prompt = gr.Textbox(label="Prompt", value="What is in the image?")
            with gr.Row():
                temp = gr.Slider(0, 1, value=0.2, step=0.05, label="temperature")
                top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="top_p")
                beams = gr.Slider(1, 4, value=1, step=1, label="num_beams")
                max_tokens = gr.Slider(32, 1024, value=512, step=32, label="max_new_tokens")
            btn = gr.Button("Run")
            out_text = gr.Textbox(label="Response")
            out_time = gr.Textbox(label="Latency")
            btn.click(infer_image, inputs=[in_image, in_prompt, temp, top_p, beams, max_tokens], outputs=[out_text, out_time])

        with gr.Tab("3D Scene"):
            scene_path = gr.Textbox(label="Scene path", value="/workspace/data/demo/scannet/scene0382_01")
            scene_prompt = gr.Textbox(label="Prompt", value="Describe the scene.")
            with gr.Row():
                temp2 = gr.Slider(0, 1, value=0.2, step=0.05, label="temperature")
                top_p2 = gr.Slider(0, 1, value=0.9, step=0.05, label="top_p")
                beams2 = gr.Slider(1, 4, value=1, step=1, label="num_beams")
                max_tokens2 = gr.Slider(32, 1024, value=512, step=32, label="max_new_tokens")
            btn2 = gr.Button("Run 3D")
            out_text2 = gr.Textbox(label="Response")
            out_time2 = gr.Textbox(label="Latency")
            btn2.click(infer_scene, inputs=[scene_path, scene_prompt, temp2, top_p2, beams2, max_tokens2], outputs=[out_text2, out_time2])

    app_fastapi = FastAPI()
    gr.mount_gradio_app(app_fastapi, demo, path="/")
    return app_fastapi
