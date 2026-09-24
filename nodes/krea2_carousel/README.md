# NodoForgeLab2 — Krea2 Carousel Studio v2.3

Share package for Jnk. Contains the Beta custom-node source and a cleaned workflow.

## Install

1. Copy `custom_nodes/NodoForgeLab2_Beta` into your `ComfyUI/custom_nodes` folder.
2. Install its `requirements.txt` using the same Python environment that runs ComfyUI.
   Windows portable example, from the folder containing `python_embeded` and `ComfyUI`:

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\NodoForgeLab2_Beta\requirements.txt
   ```

3. Install the separate packs listed below, restart ComfyUI, then hard-refresh the browser.
4. Load the JSON in `workflows/`, select your installed model files, and load YOUR master image.
   The master selection and reference batch intentionally start empty.

This ZIP contains Python/JavaScript source only. Model weights, LoRA files, photos,
API keys, local logs and third-party node packs are not included.

## Workflow dependencies

The Beta package loads independently, but the supplied full workflow also uses:

- **ComfyUI core**, with `comfy_api.latest` and Krea2 support.
- **comfyui-krea2edit**: `Krea2EditGroundedEncode` and `Krea2EditModelPatch`.
  Tested with installed version 1.2.5.
- **rgthree-comfy**: `Power Lora Loader (rgthree)`.
- **ComfyUI-KJNodes**: `PathchSageAttentionKJ` (this spelling is the node's internal ID).
- **nodeforge-lab**: `NodeForge_Image_BatchLoader` and its upload/view endpoints.
  Obtain this pack separately; it is not part of the Beta package. It supplies the
  optional reference-image uploader. If unavailable, remove that uploader node
  and its connection to `pose_reference`; Master Auto and Vibe List still work.
  Reference Copy requires the compatible uploader.
- **SageAttention** in your ComfyUI Python environment when using the Sage patch.
  The workflow retains `sage_attention=auto`; disable that patch if your environment
  does not support it.

## Required model selections

The workflow retains the tested filenames; select equivalent files installed on your computer:

- Diffusion model: `krea2_turbo_fp8_scaled.safetensors`.
- Text/vision encoder: `qwen3vl_4b_bf16.safetensors`, CLIP type `krea2`.
- VAE: `qwen_image_vae.safetensors`.
- Required editing LoRA: `krea2_identity_edit_v1_2_r128.safetensors`, strength 1.0.

The required Identity Edit LoRA is retained. The requested personal LoRA entry was
removed completely from the Power LoRA stack. Other optional rows remain OFF;
select your own files before enabling any of them. No optional LoRA is needed to
try the base workflow. A compatible character LoRA can be enabled with its trigger
in the Director.

## Generate a carousel

Keep one master image as the source for the same person, clothes, location and lighting.
The 2 MP scaler preserves its aspect ratio and rounds dimensions to multiples of 32.

- **Master Auto**: analyzes the master and creates 1–8 new pose instructions.
- **Reference Copy**: upload pose examples into the Batch Loader; choose Simple or
  Detailed, then press Run pose analysis. Each image supplies one pose, hand placement
  and camera/framing instruction. It does not replace the master image.
- **Vibe List**: creates poses from written creative direction without an image.

Choose **Transformers local**, **Ollama**, **llama.cpp**, **Gemini API** or **Grok API**.
The default local VLM is `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated`; its model
download is separate from the Krea2 encoder. For local servers, select a vision-capable
model and the correct local endpoint. For Grok, enter your own private key, click
Models, and select a model available to your account. Cloud analysis sends the selected
reference/master to that provider and uses your API account.

Press **Prepare pose list**. **Review first** lets you edit the list before Queue carousel.
**Queue automatically** submits the workflow once after the full list is ready. One
workflow execution renders all active slides, each starting from the same master.
Opening the workflow or uploading a photo does not start inference or rendering.
Keep Unload local model enabled to release VRAM before Krea2 runs.

Saved prepared lists can be rendered again without another pose-model call. Prepare
again after changing the master, references or generation settings. If you replace an
image on disk under the exact same filename, manually prepare again.

## Validation and limits

Source version: 53 Python tests and 20 JavaScript tests passed. Real local VLM analysis
produced four poses. A separate automatic two-slide test completed with Krea2 Turbo,
Identity Edit and SageAttention, producing two 1248 x 1664 images without execution
errors. The personal character LoRA was OFF in that test.

Grok/Gemini adapters were tested with simulated provider responses for this release;
the new mode was not tested with live paid APIs. Model output still needs visual review:
the tested selfie used a wider framing than requested. Continuity instructions do not
guarantee identical pixels or perfect likeness.

The cleaned distribution was checked for ZIP integrity, Python syntax, workflow links,
empty personal-image selections, absence of the removed LoRA entry and common secret
formats. `CHECKSUMS.sha256` records every payload file.
