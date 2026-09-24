"""Backend VLM local aislado (Onyx Krea2 Carousel, porte de NodoForgeLab2 Beta)."""

from __future__ import annotations

import gc
import json
import re
import threading
from pathlib import Path


DEFAULT_MODEL = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"

_LOCK = threading.Lock()
_STATE = {
    "model": None,
    "processor": None,
    "model_id": None,
    "four_bit": None,
    "attention": None,
}
STATUS = {"stage": "idle", "detail": "", "model_id": DEFAULT_MODEL}


def _set_status(stage: str, detail: str = "") -> None:
    STATUS["stage"] = stage
    STATUS["detail"] = detail


def is_loaded() -> bool:
    return _STATE["model"] is not None


def _release_unlocked() -> None:
    _STATE.update(
        model=None,
        processor=None,
        model_id=None,
        four_bit=None,
        attention=None,
    )
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload() -> None:
    with _LOCK:
        _release_unlocked()
        _set_status("idle")


def _snapshot_has_complete_weights(snapshot_path: Path) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot_path / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set((payload.get("weight_map") or {}).values())
        except (OSError, ValueError, TypeError):
            return False
        return bool(shards) and all((snapshot_path / shard).is_file() for shard in shards)
    return any(snapshot_path.glob("*.safetensors")) or any(snapshot_path.glob("pytorch_model*.bin"))


def list_cached_models() -> list[dict]:
    """Devuelve únicamente checkpoints visuales completos soportados por Transformers."""
    from huggingface_hub import scan_cache_dir

    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

        supported_types = set(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys())
        supported_architectures = set()
        for names in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
            if isinstance(names, str):
                supported_architectures.add(names)
            else:
                supported_architectures.update(names)
    except Exception:
        supported_types = {
            "gemma3",
            "gemma3n",
            "glm4v",
            "idefics2",
            "idefics3",
            "internvl",
            "llava",
            "llava_next",
            "llava_onevision",
            "mistral3",
            "mllama",
            "qwen2_5_vl",
            "qwen2_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "smolvlm",
        }
        supported_architectures = set()

    found = []
    for repo in scan_cache_dir().repos:
        if getattr(repo, "repo_type", None) != "model":
            continue
        revisions = sorted(
            getattr(repo, "revisions", ()),
            key=lambda revision: getattr(revision, "last_modified", 0) or 0,
            reverse=True,
        )
        for revision in revisions:
            snapshot = Path(revision.snapshot_path)
            config_path = snapshot / "config.json"
            if not config_path.is_file() or not _snapshot_has_complete_weights(snapshot):
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            model_type = str(config.get("model_type") or "").strip()
            architectures = [str(item) for item in (config.get("architectures") or [])]
            if model_type not in supported_types and not any(
                item in supported_architectures for item in architectures
            ):
                continue
            found.append(
                {
                    "id": repo.repo_id,
                    "model_type": model_type,
                    "architecture": architectures[0] if architectures else "",
                    "size_bytes": int(getattr(revision, "size_on_disk", 0) or 0),
                }
            )
            break
    return sorted(found, key=lambda item: item["id"].casefold())


def _flash_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _resolve_attention(choice: str, quantized: bool) -> str:
    if quantized:
        return "sdpa"
    if choice in {"sdpa", "eager"}:
        return choice
    if choice == "flash_attention_2":
        return "flash_attention_2" if _flash_available() else "sdpa"
    return "flash_attention_2" if _flash_available() else "sdpa"


def _load_error(model_id: str, exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    network_markers = (
        "couldn't connect",
        "could not connect",
        "connection error",
        "max retries",
        "name resolution",
        "timed out",
        "offline mode",
        "localentrynotfound",
    )
    if any(marker in lower for marker in network_markers):
        return (
            f"No se pudo descargar '{model_id}'. Comprueba Internet, HF_ENDPOINT y la caché local. "
            f"Error original: {message}"
        )
    return (
        f"No se pudo cargar '{model_id}' como VLM compatible. Usa un checkpoint multimodal "
        f"soportado nativamente por Transformers. Error original: {message}"
    )


def _from_pretrained(model_class, model_id: str, kwargs: dict):
    try:
        return model_class.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise
        legacy = dict(kwargs)
        legacy["torch_dtype"] = legacy.pop("dtype")
        return model_class.from_pretrained(model_id, **legacy)


def _load(model_id: str, four_bit: bool, attention: str) -> None:
    if (
        _STATE["model"] is not None
        and _STATE["model_id"] == model_id
        and _STATE["four_bit"] == four_bit
        and _STATE["attention"] == attention
    ):
        return

    _release_unlocked()
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as VisionModel
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as VisionModel
        except ImportError as exc:
            raise RuntimeError(
                "La versión instalada de Transformers no incluye AutoModelForImageTextToText "
                "ni Qwen3VLForConditionalGeneration. Actualiza Transformers."
            ) from exc

    if four_bit and not torch.cuda.is_available():
        raise RuntimeError("La carga en 4-bit necesita una GPU NVIDIA con CUDA.")

    dtype = torch.float32
    device_map = "cpu"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"

    quantization = None
    if four_bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "No se pudo inicializar bitsandbytes. Instálalo o desactiva 4-bit. "
                f"Error original: {exc}"
            ) from exc

    attention_impl = _resolve_attention(attention, quantization is not None)
    kwargs = {
        "device_map": device_map,
        "dtype": dtype,
        "trust_remote_code": False,
        "attn_implementation": attention_impl,
    }
    if quantization is not None:
        kwargs.pop("dtype")
        kwargs["quantization_config"] = quantization

    _set_status("loading", f"Cargando {model_id} ({attention_impl})…")
    try:
        try:
            model = _from_pretrained(VisionModel, model_id, kwargs)
        except Exception as exc:
            if attention_impl != "flash_attention_2":
                raise
            kwargs["attn_implementation"] = "sdpa"
            attention_impl = "sdpa"
            model = _from_pretrained(VisionModel, model_id, kwargs)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
    except Exception as exc:
        model = processor = None
        _release_unlocked()
        raise RuntimeError(_load_error(model_id, exc)) from exc

    model.eval()
    _STATE.update(
        model=model,
        processor=processor,
        model_id=model_id,
        four_bit=four_bit,
        attention=attention,
    )
    _set_status("ready", f"{model_id} · {attention_impl}")


def _apply_template(processor, messages):
    options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **options)
    except (TypeError, ValueError):
        return processor.apply_chat_template(messages, **options)


def _input_device(model):
    device_map = getattr(model, "hf_device_map", None) or {}
    candidates = []
    for value in device_map.values():
        if isinstance(value, int):
            candidates.append(f"cuda:{value}")
        elif isinstance(value, str) and value not in {"disk", "meta"}:
            candidates.append(value)
    for prefix in ("cuda", "xpu", "mps", "cpu"):
        for value in candidates:
            if value.startswith(prefix):
                return value
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return model.device


def generate_text(
    system_prompt,
    user_prompt,
    pil_images=None,
    image_labels=None,
    model_id=None,
    four_bit=False,
    unload_after=True,
    attn="auto",
    temperature=0.4,
    max_new_tokens=768,
    seed=0,
):
    """Genera texto con varias imágenes; toda la operación GPU queda serializada."""
    import torch

    model_id = (model_id or DEFAULT_MODEL).strip()
    images = [image for image in (pil_images or []) if image is not None]
    labels = list(image_labels or [])
    STATUS["model_id"] = model_id

    with _LOCK:
        model = processor = inputs = generated = None
        try:
            _load(model_id, bool(four_bit), attn)
            model = _STATE["model"]
            processor = _STATE["processor"]
            _set_status("generating", "Generando prompt dirigido…")

            content = []
            for index, image in enumerate(images, start=1):
                label = labels[index - 1] if index <= len(labels) else f"IMAGE {index}"
                content.extend(
                    (
                        {"type": "text", "text": f"{label}:"},
                        {"type": "image", "image": image},
                    )
                )
            content.append({"type": "text", "text": user_prompt})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": content},
            ]
            try:
                inputs = _apply_template(processor, messages)
            except (TypeError, ValueError):
                merged = [{"type": "text", "text": system_prompt + "\n\n"}, *content]
                inputs = _apply_template(processor, [{"role": "user", "content": merged}])
            inputs = inputs.to(_input_device(model))

            normalized_seed = int(seed) % (2**32)
            torch.manual_seed(normalized_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(normalized_seed)

            sampling = float(temperature) > 0.01
            generation_args = {
                "max_new_tokens": max(32, min(int(max_new_tokens), 2048)),
                "do_sample": sampling,
            }
            if sampling:
                generation_args.update(
                    temperature=max(0.01, min(float(temperature), 2.0)),
                    top_p=0.9,
                )
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_args)
            trimmed = generated[:, inputs["input_ids"].shape[1] :]
            text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            _set_status("ready")
            return text.strip()
        except Exception as exc:
            _set_status("error", str(exc))
            raise
        finally:
            model = processor = inputs = generated = None
            if unload_after:
                _release_unlocked()
                if STATUS["stage"] != "error":
                    _set_status("idle")
