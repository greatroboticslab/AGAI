"""
MiniGPT-v2 model loading, prompt construction, and inference.

This module must be imported from the minigptv conda environment
(Python 3.9, with minigpt4, omegaconf, torch, etc. installed).
"""

import importlib
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from .config import MINIGPT_EVAL_CONFIG, PLANT_PARTS

# ── Lazy MiniGPT registration ────────────────────────────────────────────────

_REGISTERED = False


def _ensure_registered():
    """Import MiniGPT subpackages to trigger @registry.register_* decorators."""
    global _REGISTERED
    if _REGISTERED:
        return
    for mod in [
        "minigpt4.datasets.builders",
        "minigpt4.models",
        "minigpt4.processors",
        "minigpt4.runners",
        "minigpt4.tasks",
    ]:
        importlib.import_module(mod)
    _REGISTERED = True


# ── Model loading ────────────────────────────────────────────────────────────

def load_minigpt(gpu_id=0):
    """Load MiniGPT-v2 and return (chat, conv_template).

    The chat object wraps image upload, encoding, prompt injection, and
    generation.  conv_template is copied per-image to keep conversations
    independent.
    """
    _ensure_registered()

    import torch
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import (
        Conversation, SeparatorStyle, Chat,
    )

    class _Args:
        cfg_path = MINIGPT_EVAL_CONFIG
        options = None

    _Args.gpu_id = gpu_id
    cfg = Config(_Args())

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.load_checkpoint(cfg.model_cfg.ckpt)
    model = model.eval()

    # Patch out cache_position kwarg (compatibility shim for newer transformers)
    for module in model.modules():
        f = getattr(module, "forward", None)
        if f is None or getattr(f, "_drops_cachepos", False):
            continue
        def _wrapped(*args, __orig=f, **kwargs):
            kwargs.pop("cache_position", None)
            return __orig(*args, **kwargs)
        _wrapped._drops_cachepos = True
        try:
            module.forward = _wrapped
        except Exception:
            pass

    try:
        vp_cfg = (getattr(cfg.preprocess_cfg.vis_processor, "eval", None)
                  or cfg.preprocess_cfg.vis_processor.train)
        vis_processor = registry.get_processor_class(vp_cfg.name).from_config(vp_cfg)
    except Exception:
        vis_processor = registry.get_processor_class(
            "blip2_image_eval"
        ).from_config({"image_size": 448})

    chat = Chat(model, vis_processor, device=device)

    conv_template = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )
    return chat, conv_template


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(disease_label, detected_parts=None):
    """Build (system_prompt, user_prompt) for MiniGPT.

    When detected_parts is provided (grounding mode), the prompt constrains
    MiniGPT to only describe those parts.  When None, MiniGPT describes
    any parts freely.

    Returns:
        (system_prompt: str, user_prompt: str)
    """
    disease_name = disease_label.replace("_", " ").title()

    if detected_parts:
        visible = sorted(detected_parts.keys())
        hidden = sorted(PLANT_PARTS - set(detected_parts.keys()))
        visible_csv = ", ".join(visible)
        hidden_csv = ", ".join(hidden)

        system = (
            f"<<SYS>>You are a plant diagnostician. "
            f"The diagnosis has already been determined: {disease_name}\n"
            f"A vision-based detector has confirmed which plant parts are "
            f"visible in this image.\n"
            f"VISIBLE (describe these): {visible_csv}.\n"
            f"NOT VISIBLE (do NOT mention these at all): {hidden_csv}.\n"
            f"You must ONLY describe symptoms on the VISIBLE parts listed above. "
            f"Do NOT name, reference, or explain anything involving "
            f"{hidden_csv} — not even as causes, locations, or in advice.\n"
            f"The green sepals on top of a strawberry are the calyx, "
            f"part of the fruit, not leaves.<</SYS>>"
        )
        user = (
            f"Analyze this strawberry plant image. "
            f"The diagnosis is {disease_label.replace('_', ' ')}. "
            f"VISIBLE parts: {visible_csv}. "
            f"NOT VISIBLE (do not mention): {hidden_csv}. "
            f"Describe symptoms ONLY on the visible parts."
        )
    else:
        system = (
            f"<<SYS>>You are a plant diagnostician. "
            f"The diagnosis has already been determined: {disease_name}\n"
            f"Examine the image and explain why this diagnosis is correct.\n"
            f"Describe the visible symptoms on specific plant parts "
            f"(leaves, fruit, flowers, stems, roots, soil).\n"
            f"Be detailed about which plant parts show symptoms.<</SYS>>"
        )
        user = (
            f"Analyze this strawberry plant image. "
            f"The diagnosis is {disease_label.replace('_', ' ')}. "
            f"Describe the visible symptoms on each plant part you can see."
        )

    return system, user


# ── Inference ────────────────────────────────────────────────────────────────

def run_minigpt(chat, conv_template, image_path, disease_label,
                detected_parts=None):
    """Generate a diagnostic response for a single image.

    Args:
        chat:            Chat object from load_minigpt().
        conv_template:   Conversation template (will be copied, not mutated).
        image_path:      Path to the image file.
        disease_label:   e.g. "white_mold".
        detected_parts:  dict {part: confidence} or None (ungrounded mode).

    Returns:
        str  — MiniGPT's generated text.
    """
    from PIL import Image as PILImage

    system, user = build_prompt(disease_label, detected_parts)

    conv = conv_template.copy()
    conv.system = system

    img_list = []
    img = PILImage.open(image_path).convert("RGB")
    chat.upload_img(img, conv, img_list)
    chat.encode_img(img_list)

    chat.ask(user, conv)
    response = chat.answer(
        conv, img_list, temperature=0.6, max_new_tokens=512, max_length=2000,
    )[0]
    return response
