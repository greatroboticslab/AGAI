"""Vision backbones shared by cluster/classifier/probe_backbones.py and cluster/classifier/train_backbone.py: local weight paths, input
resolution, normalization, and one feature function per family (all run offline from /projects/$USER/agai/models)."""
import json
import os

import torch

M = f"/projects/{os.environ['USER']}/agai/models"
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
BACKBONES = {
    "dinov2-giant": dict(kind="hf_vit", path=f"{M}/dinov2-giant", res=448, norm=IMNET),
    "dinov2-large": dict(kind="hf_vit", path=f"{M}/dinov2-large", res=448, norm=IMNET),
    "siglip2-so400m-512": dict(kind="hf_siglip", path=f"{M}/siglip2-so400m-patch16-512", res=512, norm=HALF),
    "convnextv2-large-384": dict(kind="hf_pooled", path=f"{M}/convnextv2-large-22k-384", res=384, norm=IMNET),
    "eva02-large-448": dict(kind="timm", arch="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", res=448, norm=CLIP,
                            path=f"{M}/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k/model.safetensors"),
    "bioclip-2": dict(kind="open_clip", arch="ViT-L-14", path=f"{M}/bioclip-2/open_clip_model.safetensors",
                      res=224, norm=CLIP),
    "bioclip-2.5-h14": dict(kind="open_clip", arch="ViT-H-14", res=224, norm=CLIP,
                            path=f"{M}/bioclip-2.5-vith14/open_clip_model.safetensors"),
    "pe-core-g14-448": dict(kind="timm", arch="vit_pe_core_gigantic_patch14_448.fb", res=448, norm=HALF,
                            path=f"{M}/vit_pe_core_gigantic_patch14_448.fb/model.safetensors"),
    "plantnet-dinov2-b": dict(kind="timm", arch="vit_base_patch14_reg4_dinov2.lvd142m", res=518, norm=IMNET,
                              path=f"{M}/vit_base_patch14_reg4_dinov2_lvd142m.pc24_all/model.safetensors"),
    "dinov3-vitl16": dict(kind="hf_vit", path=f"{M}/dinov3-vitl16-pretrain-lvd1689m", res=448, norm=IMNET),
    "dinov3-vith16plus": dict(kind="hf_vit", path=f"{M}/dinov3-vith16plus-pretrain-lvd1689m", res=448, norm=IMNET),
    "dinov3-vit7b16": dict(kind="hf_vit", path=f"{M}/dinov3-vit7b16-pretrain-lvd1689m", res=448, norm=IMNET),
}


def load(spec, dtype=torch.bfloat16):
    """(module, feats) where feats(x) -> [B, D] pooled features. Sets spec['norm'] from the checkpoint's own
    preprocessing config when it has one."""
    k = spec["kind"]
    if k in ("hf_vit", "hf_siglip", "hf_pooled"):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(spec["path"], dtype=dtype)
        if k == "hf_siglip":
            m = m.vision_model
    elif k == "timm":   # by hand, so classifier heads (e.g. PlantNet's 7,806 species) are dropped cleanly
        import timm
        from safetensors.torch import load_file
        m = timm.create_model(spec["arch"], pretrained=False, num_classes=0)
        own, sd = m.state_dict(), load_file(spec["path"])
        keep = {n: v for n, v in sd.items() if n in own and v.shape == own[n].shape}
        missing = [n for n in own if n not in keep]
        if missing:
            raise RuntimeError(f"{len(missing)} weights missing from {spec['path']}, e.g. {missing[:4]}")
        m.load_state_dict(keep)
        spec["norm"] = (tuple(m.pretrained_cfg["mean"]), tuple(m.pretrained_cfg["std"]))
        m = m.to(dtype)
    else:
        import open_clip
        cfg_path = os.path.join(os.path.dirname(spec["path"]), "open_clip_config.json")
        if os.path.exists(cfg_path):
            pp = json.load(open(cfg_path)).get("preprocess_cfg", {})
            if "mean" in pp:
                spec["norm"] = (tuple(pp["mean"]), tuple(pp["std"]))
        m = open_clip.create_model(spec["arch"], pretrained=spec["path"]).visual.to(dtype)

    def feats(x):
        if k == "hf_vit":   # CLS + mean of patch tokens (register tokens skipped)
            h = m(pixel_values=x).last_hidden_state
            n_reg = getattr(m.config, "num_register_tokens", 0) or 0
            return torch.cat([h[:, 0], h[:, 1 + n_reg:].mean(1)], 1)
        if k == "hf_siglip":
            o = m(pixel_values=x)
            return torch.cat([o.pooler_output, o.last_hidden_state.mean(1)], 1)
        if k == "hf_pooled":
            return m(pixel_values=x).pooler_output.flatten(1)
        return m(x)   # timm (num_classes=0: pooled pre-logits) / open_clip visual tower (projected embedding)
    return m, feats


def enable_grad_checkpointing(m):
    for name in ("gradient_checkpointing_enable", "set_grad_checkpointing"):
        fn = getattr(m, name, None)
        if fn is not None:
            try:
                fn() if name == "gradient_checkpointing_enable" else fn(True)
                return True
            except Exception:
                pass
    return False
