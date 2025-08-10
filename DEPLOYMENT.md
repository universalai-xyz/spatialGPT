Deployment to Modal (GPU) for LLaVA-3D

Prerequisites
- Python 3.10+
- Modal account and CLI installed: pip install modal
- Modal token: modal token set
- Optional: Hugging Face token for faster model downloads (configure a Modal secret named "huggingface" with HUGGINGFACE_HUB_TOKEN), optional OpenAI key as a secret named "openai" if moderation is later enabled
- Ensure repository root contains modal_app.py and deploy.sh

GPU and Model
- Default GPU: A10G (24GB)
- Model: ChaimZhu/LLaVA-3D-7B (downloaded from Hugging Face)
- Precision: bfloat16 by default; override via env SPATIALGPT_PRECISION
- If you need a different GPU, set env SPATIALGPT_MODAL_GPU accordingly (e.g., T4). A10G is recommended.

Secrets
- Create Modal secrets:
  - huggingface: include HUGGINGFACE_HUB_TOKEN=your_token if needed
  - openai (optional): include OPENAI_API_KEY=your_key if needed later
- Example:
  - modal secret create huggingface --env HUGGINGFACE_HUB_TOKEN=hf_xxx
  - modal secret create openai --env OPENAI_API_KEY=sk-...

Deploy
- From repo root:
  - bash deploy.sh
- The CLI will print the deployment result and public URL when ready.
- To follow logs:
  - modal logs -f spatialgpt-llava3d.web

Public URL format
- Modal prints a URL similar to: https://modal-labs-example-...modal.run
- Use the printed URL from deploy output

Gradio App
- Tabs:
  - Image: upload an image and enter a prompt
  - 3D Scene: provide a server-side path containing posed RGB-D for a scene
- Latency is displayed per request. Aim for under 10 seconds per image on A10G.

Usage and Costs
- Modal bills for GPU time and storage for volumes used as cache
- The app uses volumes for Hugging Face cache and data to avoid re-downloading weights
- Monitor logs and usage in Modal dashboard and with modal logs -f

Advanced Configuration
- Model selection: set SPATIALGPT_MODEL (default ChaimZhu/LLaVA-3D-7B)
- Precision: set SPATIALGPT_PRECISION to fp16 or bf16
- GPU: set SPATIALGPT_MODAL_GPU to A10G or T4

Smoke Test
- After deployment, open the public URL
- Test with three images of varying complexity
- Verify responses are returned and latency is under 10 seconds per image
