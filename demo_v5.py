
#William Starks - Plant Diagnostic MiniGPT, derived from demo_v4.py and modified into a resnet50-wired strawberry pathologist. WIP
#Gus Marcum - Collaborator: debugging, and system improvements

#Added CSS to the gradio app for user interface improvements

# Suppress known deprecation warnings from third-party libraries (timm, tensorflow)
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARNING messages
warnings.filterwarnings("ignore", message="Importing from timm.models.hub is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")

#Standard library imports
import argparse
import re
import sys
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import networkx as nx
import plotly.graph_objects as go
from PIL import Image

# SERPAPI Configuration (optional: only used in Enhanced mode)
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
SERPAPI_AVAILABLE = False
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = bool(SERP_API_KEY)
except ImportError:
    SERPAPI_AVAILABLE = False
    print("Warning: serpapi not available, web search features disabled")

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from resnet_classifier import load_resnet, diagnose_or_none

# Import RAG retriever for disease knowledge
try:
    from knowledge_graph.rag_retriever import DiseaseRAG
    from knowledge_graph.qa_retriever import DiseaseQA
    disease_rag = DiseaseRAG()
    disease_qa = DiseaseQA()
    RAG_AVAILABLE = True
    print("[RAG] Disease knowledge base loaded successfully")
except Exception as e:
    RAG_AVAILABLE = False
    disease_rag = None
    disease_qa = None
    print(f"[RAG] Warning: RAG not available: {e}")

# Import RF-DETR grounding detector for constraining MiniGPT output
try:
    from grounding.detector import run_rfdetr
    from grounding.config import PLANT_PARTS
    RFDETR_AVAILABLE = True
    print("[RF-DETR] Grounding detector available")
except Exception as e:
    RFDETR_AVAILABLE = False
    print(f"[RF-DETR] Warning: Grounding not available: {e}")

# Configure logging
logging.basicConfig(filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)


# Configuration path
DEFAULT_CFG_PATH = "eval_configs/minigptv2_eval.yaml"

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Plant Diagnostic System")
    parser.add_argument("--cfg-path", default=DEFAULT_CFG_PATH, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--options", default=None, help="additional options to override the configuration.")
    args = parser.parse_args()
    return args

args = parse_args()

# --- Validate configuration file ---
cfg_path = Path(args.cfg_path)
if not cfg_path.exists():
    print(f"\n[ERROR] Configuration file not found: {args.cfg_path}")
    print(f"        Please ensure the config file exists at the specified path.")
    print(f"        Default: {DEFAULT_CFG_PATH}\n")
    sys.exit(1)

# --- ResNet anchor (required) ---
_RESNET_MODEL = None

def _get_resnet():
    global _RESNET_MODEL
    if _RESNET_MODEL is None:
        model_paths = [
            Path(__file__).parent / "plant_diagnostic" / "models" / "resnet_strawberry.pth",
            Path("plant_diagnostic/models/resnet_strawberry.pth"),
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = str(path)
                break
        
        if model_path is None:
            raise FileNotFoundError(f"ResNet model not found at: {model_paths}")
        
        _RESNET_MODEL = load_resnet(model_path)
    return _RESNET_MODEL

# Load ResNet at startup (required)
try:
    _get_resnet()
    print("[ResNet] Model loaded successfully")
except Exception as e:
    print(f"\n[ERROR] Failed to load ResNet classifier: {e}")
    print(f"        Ensure resnet_strawberry.pth exists in plant_diagnostic/models/\n")
    sys.exit(1)

# --- Fixed-label helpers ---
_CLASS_THRESH = {
    "healthy":      0.40, 
    "overwatered":  0.60,
    "root_rot":     0.65,
    "drought":      0.70,
    "frost":        0.75,
    "gray_mold":    0.60,  
    "white_mold":   0.60,  
}

_CANON_LABEL_MAP = {
    "healthy": "healthy",
    "overwatered": "overwatering", 
    "root_rot": "root rot",
    "drought": "drought",
    "frost": "frost injury",
    "gray_mold": "gray mold",  
    "white_mold": "white mold",  
}

# Global acceptance defaults (tune if needed)
_DEFAULT_UNKNOWN_THRESH = 0.55     # minimum p1 to accept top-1
_MIN_MARGIN_OVER_TOP2   = 0.08     # only used if p2 provided

def _accept_label(pred) -> str:
    """
    Decide whether to accept ResNet's top prediction or return 'unknown'.
    pred fields expected:
        label: str (top-1 class name)
        p1: float (top-1 prob)
        p2: float (optional, top-2 prob)
    """
    if not pred:
        print("[ResNet] No prediction returned")
        return "unknown"

    lbl_raw = str(pred.get("label", "")).strip()
    lbl = lbl_raw.lower()
    p1 = float(pred.get("p1", 0.0))
    p2 = float(pred.get("p2", 0.0)) if "p2" in pred else 0.0

    # Map to canon label (what UI shows)
    canon = _CANON_LABEL_MAP.get(lbl, lbl)

    # Per-class threshold override falls back to global default
    thr = float(_CLASS_THRESH.get(lbl, _DEFAULT_UNKNOWN_THRESH))
    margin_ok = (p2 == 0.0) or ((p1 - p2) >= _MIN_MARGIN_OVER_TOP2)

    print(f"[ResNet] Label: {lbl} -> {canon}, p1={p1:.3f}, p2={p2:.3f}, thr={thr:.2f}, margin_ok={margin_ok}")

    if p1 < thr:
        print("[ResNet] BELOW THRESH -> unknown")
        return "unknown"
    if not margin_ok:
        print("[ResNet] SMALL MARGIN OVER TOP2 -> unknown")
        return "unknown"

    # Accept only known labels; unknown string falls back
    return _CANON_LABEL_MAP.get(lbl, "unknown")

def _postprocess_caption(text: str) -> str:
    """Light cleanup for LLM output: normalize quotes and spacing, keep content intact."""
    if not text:
        return ""
    t = text.strip()

    # Normalize curly quotes -> straight quotes
    t = (t.replace("\u201c", '"').replace("\u201d", '"')   # " "
         .replace("\u2018", "'").replace("\u2019", "'"))   # '  '

    # Remove simple HTML-ish tags and zero-width chars
    t = re.sub(r"</?[^>\s]{1,32}>?", "", t)
    t = t.replace("\u200b", "").replace("\\n", "\n")
    t = t.replace("<", "").replace(">", "")  # defensive

    # Formatting touch-ups
    t = re.sub(r"\*\s*", "• ", t)                 # '* item' -> '• item'
    t = re.sub(r"\n{3,}", "\n\n", t)              # collapse 3+ newlines to 2
    t = re.sub(r"\n\n+", "\n\n", t)               # ensure max 1 blank line
    return t

# ── RF-DETR helpers ───────────────────────────────────────────────────────────

_BBOX_COLORS = {
    "flower": "#FF6B9D", "fruit": "#FF4444", "leaf": "#48BB78",
    "root": "#B07D4B", "soil": "#8B6914", "stem": "#38B2AC",
}

_CANON_TO_SNAKE = {
    "healthy": "healthy", "overwatering": "overwatered", "root rot": "root_rot",
    "drought": "drought", "frost injury": "frost", "gray mold": "gray_mold",
    "white mold": "white_mold",
}

def _run_rfdetr_on_image(img_path):
    """Run RF-DETR detection, returns (detected_parts dict, all_detections list)."""
    if not RFDETR_AVAILABLE:
        return {}, []
    try:
        results = run_rfdetr([img_path])
        if not results:
            return {}, []
        det = results.get(img_path, {})
        return det.get("detected_parts", {}), det.get("all_detections", [])
    except Exception as e:
        print(f"[RF-DETR] Detection failed: {e}")
        return {}, []


def _draw_bboxes(pil_img, detections):
    """Draw bounding boxes on a copy of the image. Returns annotated PIL Image."""
    from PIL import ImageDraw, ImageFont
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        cls = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = _BBOX_COLORS.get(cls, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls} {conf:.0%}"
        tx, ty = x1, max(y1 - 18, 0)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill="white", font=font)
    return img


# Load configuration
cfg = Config(args)
if args.options is not None:
    cfg.update_with_str(args.options)

# Set device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)

# Initialize model
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.load_checkpoint(cfg.model_cfg.ckpt)
model = model.eval()
print(f"[ckpt] using: {cfg.model_cfg.ckpt}")

# Model configuration patches
try:
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
        try:
            model.generation_config.cache_implementation = "static"
        except Exception:
            pass
except Exception:
    pass

try:
    if hasattr(model, "config"):
        model.config.use_cache = False
except Exception:
    pass

def _patch_drop_cachepos_all(root):
    wrapped = 0
    for module in root.modules():
        f = getattr(module, "forward", None)
        if f is None:
            continue
        if getattr(f, "_drops_cachepos", False):
            continue

        def wrapped_forward(*args, __orig=f, **kwargs):
            kwargs.pop("cache_position", None)
            return __orig(*args, **kwargs)

        setattr(wrapped_forward, "_drops_cachepos", True)
        try:
            module.forward = wrapped_forward
            wrapped += 1
        except Exception:
            pass
    print(f"[patch] drop(cache_position): wrapped {wrapped} module.forward funcs")

_patch_drop_cachepos_all(model)

# Initialize visual processor
try:
    vis_processor_cfg = getattr(cfg.preprocess_cfg.vis_processor, "eval", None) or cfg.preprocess_cfg.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
except Exception:
    vis_processor = registry.get_processor_class('blip2_image_eval').from_config({'image_size': 448})

# Chat patches
try:
    _ORIG_ANSWER_PREPARE = Chat.answer_prepare
    def _answer_prepare_nocache(self, *args, **kwargs):
        kwargs.pop("use_cache", None)
        return _ORIG_ANSWER_PREPARE(self, *args, **kwargs)
    Chat.answer_prepare = _answer_prepare_nocache
    print("[patch] Chat.answer_prepare patched to drop 'use_cache'")
except Exception as e:
    print(f"[patch] Could not patch answer_prepare: {e}")

# Initialize chat
chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

# Define CONV_VISION
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def fetch_serp_context(query):
    """Fetch additional context from SERPAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 3,
        "gl": "us",
        "hl": "en",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        context = " ".join([result.get("snippet", "") for result in organic_results[:3]])
        return context
    except Exception as e:
        logging.error(f"SERPAPI Error: {e}")
        return ""

def _empty_fig(msg):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(
            text=msg, 
            x=0.5, 
            y=0.5, 
            showarrow=False,
            font=dict(color="#b0b0b0", size=14)
        )],
        autosize=True,
        height=620,
        margin=dict(b=30, l=20, r=20, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='#151515',
        plot_bgcolor='#151515',
    )
    return fig

def create_knowledge_graph():
    """Create the disease knowledge graph visualization using RAG knowledge base."""
    try:
        if not RAG_AVAILABLE or disease_rag is None:
            return _empty_fig("Disease knowledge base not loaded")
        
        nodes, edges = disease_rag.get_graph_data()
        
        if not nodes:
            return _empty_fig("No disease data available")
        
        # Build NetworkX graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(
                node['id'], 
                label=node['label'], 
                name=node['name'],
                severity=node.get('severity', ''),
                full_text=node.get('full_text', node['name'])
            )
        for edge in edges:
            G.add_edge(edge['start_id'], edge['end_id'], type=edge['type'])
        
        if G.number_of_nodes() == 0:
            return _empty_fig("No nodes to display.")
        
        # Use spring layout with higher k for better spacing
        k = 2.5 / max(1, G.number_of_nodes() ** 0.5)
        pos = nx.spring_layout(G, k=k, iterations=300, seed=42)
        
        # Color scheme for node types
        label_colors = {
            'Disease': '#ff6b6b',      # Red for diseases
            'Symptom': '#ffd93d',      # Yellow for symptoms
            'Treatment': '#48bb78',    # Green for treatments
            'Cause': '#b794f4',        # Purple for causes
            'Recovery': '#00d4ff',     # Cyan for recovery
        }
        
        # Severity-based sizing for disease nodes
        severity_sizes = {
            'severe': 28,
            'moderate': 24,
            'none': 20,
            'unknown': 20
        }
        
        # Edge traces by type
        def _edge_trace_for(rel_type, color, width=1.0):
            ex, ey, et = [], [], []
            for u, v, d in G.edges(data=True):
                if d.get('type') != rel_type:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                et.extend([rel_type.replace('_', ' '), rel_type.replace('_', ' '), None])
            return go.Scatter(
                x=ex, y=ey,
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=et,
                mode='lines',
                opacity=0.5
            )
        
        edge_symptom = _edge_trace_for('Shows_Symptom', '#ffd93d', 0.8)
        edge_treatment = _edge_trace_for('Treated_By', '#48bb78', 1.0)
        edge_cause = _edge_trace_for('Causes', '#b794f4', 0.8)
        edge_recovery = _edge_trace_for('Recovery_Time', '#00d4ff', 0.8)
        
        # Node trace
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            attrs = G.nodes[node_id]
            label = attrs.get('label', '?')
            name = attrs.get('name', '?')
            full_text = attrs.get('full_text', name)
            severity = attrs.get('severity', '')
            
            # Build hover text
            if label == 'Disease':
                hover = f"<b>🍓 {name}</b><br>Severity: {severity}"
            elif label == 'Symptom':
                hover = f"<b>⚠️ Symptom</b><br>{full_text}"
            elif label == 'Treatment':
                hover = f"<b>💊 Treatment</b><br>{full_text}"
            elif label == 'Cause':
                hover = f"<b>🔍 Cause</b><br>{full_text}"
            elif label == 'Recovery':
                hover = f"<b>⏱️ Recovery</b><br>{full_text}"
            else:
                hover = f"<b>{label}</b><br>{name}"
            
            node_text.append(hover)
            node_color.append(label_colors.get(label, '#718096'))
            
            # Size based on node type
            if label == 'Disease':
                node_size.append(severity_sizes.get(severity, 20))
            else:
                node_size.append(10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='rgba(255, 255, 255, 0.3)'),
            ),
        )
        
        fig = go.Figure(
            data=[edge_symptom, edge_treatment, edge_cause, edge_recovery, node_trace],
            layout=go.Layout(
                title=dict(
                    text='<b>🍓 Strawberry Disease Knowledge Graph</b>',
                    font=dict(size=18, color='#f0f0f0'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=30, l=20, r=20, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                autosize=True,
                paper_bgcolor='#151515',
                plot_bgcolor='#151515',
                hoverlabel=dict(
                    bgcolor="#1e1e1e",
                    font_size=12,
                    font_family="DM Sans, sans-serif",
                    font_color="#f0f0f0"
                ),
                height=620,
                annotations=[
                    dict(
                        text="<b>Legend:</b> 🔴 Disease  🟡 Symptom  🟢 Treatment  🟣 Cause  🔵 Recovery",
                        x=0.5, y=-0.02,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=11, color="#888888")
                    )
                ]
            )
        )
        fig.update_xaxes(scaleanchor=None, constrain=None)
        fig.update_yaxes(scaleanchor=None, constrain=None)
        return fig
    
    except Exception as e:
        print(f"[KG ERROR] {e}")
        import traceback
        traceback.print_exc()
        return _empty_fig(f"Error: {e}")

def draw_disease_detail(disease_name):
    """Draw detailed graph for a specific disease."""
    try:
        if not RAG_AVAILABLE or disease_rag is None:
            return _empty_fig("Disease knowledge base not loaded")
        
        # Normalize disease name input
        disease_key = disease_name.lower().strip().replace(" ", "_").replace("-", "_")
        
        # Map common variations
        name_map = {
            "gray": "gray_mold",
            "grey": "gray_mold",
            "gray_mold": "gray_mold",
            "grey_mold": "gray_mold",
            "botrytis": "gray_mold",
            "white": "white_mold",
            "white_mold": "white_mold",
            "sclerotinia": "white_mold",
            "root": "root_rot",
            "root_rot": "root_rot",
            "phytophthora": "root_rot",
            "frost": "frost_injury",
            "cold": "frost_injury",
            "frost_injury": "frost_injury",
            "drought": "drought",
            "dry": "drought",
            "water": "overwatering",
            "overwater": "overwatering",
            "overwatering": "overwatering",
            "healthy": "healthy",
        }
        disease_key = name_map.get(disease_key, disease_key)
        
        # Get graph data for this disease
        nodes, edges = disease_rag.get_disease_graph_data(disease_key)
        
        if not nodes:
            available = ", ".join(disease_rag.get_all_diseases())
            return _empty_fig(f"Disease '{disease_name}' not found.<br>Available: {available}")
        
        # Build NetworkX graph
        G = nx.DiGraph()
        disease_node_id = None
        for node in nodes:
            G.add_node(
                node['id'],
                label=node['label'],
                name=node['name'],
                severity=node.get('severity', ''),
                full_text=node.get('full_text', node['name'])
            )
            if node['label'] == 'Disease':
                disease_node_id = node['id']
        
        for edge in edges:
            G.add_edge(edge['start_id'], edge['end_id'], type=edge['type'])
        
        # Use shell layout with disease at center
        if disease_node_id:
            # Create radial layout with disease at center
            pos = nx.shell_layout(G, nlist=[
                [disease_node_id],  # Center
                [n for n in G.nodes() if G.nodes[n].get('label') in ['Symptom', 'Cause']],
                [n for n in G.nodes() if G.nodes[n].get('label') in ['Treatment', 'Recovery']]
            ])
        else:
            pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
        
        # Color scheme
        label_colors = {
            'Disease': '#ff6b6b',
            'Symptom': '#ffd93d',
            'Treatment': '#48bb78',
            'Cause': '#b794f4',
            'Recovery': '#00d4ff',
        }
        
        # Edge traces by type
        def _edge_trace_for(rel_type, color, width=1.2):
            ex, ey, et = [], [], []
            for u, v, d in G.edges(data=True):
                if d.get('type') != rel_type:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ex.extend([x0, x1, None])
                ey.extend([y0, y1, None])
                et.extend([rel_type.replace('_', ' '), rel_type.replace('_', ' '), None])
            return go.Scatter(
                x=ex, y=ey,
                line=dict(width=width, color=color),
                hoverinfo='text',
                text=et,
                mode='lines',
                opacity=0.6
            )
        
        edge_symptom = _edge_trace_for('Shows_Symptom', '#ffd93d', 1.0)
        edge_treatment = _edge_trace_for('Treated_By', '#48bb78', 1.2)
        edge_cause = _edge_trace_for('Causes', '#b794f4', 1.0)
        edge_recovery = _edge_trace_for('Recovery_Time', '#00d4ff', 1.0)
        
        # Node traces
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            attrs = G.nodes[node_id]
            label = attrs.get('label', '?')
            name = attrs.get('name', '?')
            full_text = attrs.get('full_text', name)
            severity = attrs.get('severity', '')
            
            # Build hover text
            if label == 'Disease':
                hover = f"<b>🍓 {name}</b><br>Severity: {severity}"
            elif label == 'Symptom':
                hover = f"<b>⚠️ Symptom</b><br>{full_text}"
            elif label == 'Treatment':
                hover = f"<b>💊 Treatment</b><br>{full_text}"
            elif label == 'Cause':
                hover = f"<b>🔍 Cause</b><br>{full_text}"
            elif label == 'Recovery':
                hover = f"<b>⏱️ Recovery</b><br>{full_text}"
            else:
                hover = f"<b>{label}</b><br>{name}"
            
            node_text.append(hover)
            node_color.append(label_colors.get(label, '#718096'))
            
            # Central disease node is larger
            if label == 'Disease':
                node_size.append(35)
            elif label in ['Treatment', 'Recovery']:
                node_size.append(14)
            else:
                node_size.append(12)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='rgba(255, 255, 255, 0.4)')
            )
        )
        
        # Get display name
        display_name = disease_rag.get_display_name(disease_key)
        
        fig = go.Figure(
            data=[edge_symptom, edge_treatment, edge_cause, edge_recovery, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f'<b>🔍 {display_name}</b>',
                    font=dict(size=18, color='#f0f0f0'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=30, l=20, r=20, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                autosize=True,
                paper_bgcolor='#151515',
                plot_bgcolor='#151515',
                hoverlabel=dict(
                    bgcolor="#1e1e1e",
                    font_size=12,
                    font_family="DM Sans, sans-serif",
                    font_color="#f0f0f0"
                ),
                height=620,
                annotations=[
                    dict(
                        text="<b>Legend:</b> 🔴 Disease  🟡 Symptom  🟢 Treatment  🟣 Cause  🔵 Recovery",
                        x=0.5, y=-0.02,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=11, color="#888888")
                    )
                ]
            )
        )
        fig.update_xaxes(scaleanchor=None, constrain=None)
        fig.update_yaxes(scaleanchor=None, constrain=None)
        return fig
    
    except Exception as e:
        print(f"[Disease Detail ERROR] {e}")
        import traceback
        traceback.print_exc()
        return _empty_fig(f"Error: {e}")

def process_chat_with_image(user_message, chatbot, chat_state, gr_img, img_list, temperature, is_enhanced=False, use_rag=False, show_bboxes=False):
    """Process chat with image analysis"""
    try:
        if gr_img is None:
            return (chatbot + [[user_message, "⚠️ Please upload an image first."]], chat_state, img_list, None)

        # Check if this is a follow-up question (user typed something AND we have previous diagnosis)
        user_question = user_message.strip() if user_message else ""
        
        # Detect if this looks like a follow-up question
        is_followup = False
        previous_diagnosis = None
        
        if user_question and chatbot and len(chatbot) > 0:
            # Check if previous response contains a diagnosis
            last_response = chatbot[-1][1] if chatbot[-1] else ""
            for disease in ["drought", "overwatering", "root_rot", "frost_injury", "gray_mold", "white_mold", "healthy"]:
                if disease.replace("_", " ") in last_response.lower() or disease in last_response.lower():
                    previous_diagnosis = disease
                    break
            
            # Check if user is asking a question (not uploading new context)
            question_indicators = ["?", "how", "why", "what", "when", "can", "will", "is it", "should", "difference", "cause", "prevent", "recover", "treat"]
            if previous_diagnosis and any(ind in user_question.lower() for ind in question_indicators):
                is_followup = True
        
        # If follow-up question, use QA-RAG directly without re-processing image
        if is_followup and previous_diagnosis and RAG_AVAILABLE and disease_qa:
            print(f"[Follow-up] Detected question about {previous_diagnosis}: {user_question[:50]}...")
            qa_result = disease_qa.answer_question(previous_diagnosis, user_question)
            
            if qa_result.get("confidence", 0) >= 0.3:
                answer = qa_result.get("answer", "")
                q_type = qa_result.get("question_type", "general")
                
                # Format the response
                response = f"**{q_type.title()} Question**\n\n{answer}"
                
                # Add suggestion for more questions
                suggestions = disease_qa.suggest_questions(previous_diagnosis)
                remaining = [s for s in suggestions if q_type not in s.lower()][:2]
                if remaining:
                    response += "\n\n---\n**Also ask:**\n"
                    for s in remaining:
                        response += f"- {s}\n"
                
                return (chatbot + [[user_question, response]], chat_state, img_list, None)
            else:
                # Low confidence - fall through to full analysis
                print(f"[Follow-up] Low confidence ({qa_result.get('confidence', 0)}), falling back to full analysis")

        # Fresh conversation + encode image (for initial diagnosis or unclear follow-up)
        chat_state = CONV_VISION.copy()
        img_list = []
        detected_parts, all_detections = {}, []
        chat.upload_img(gr_img, chat_state, img_list)
        chat.encode_img(img_list)

        # Lightweight ResNet pass (just to get a label + confidence)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img_path = tmp_file.name
        gr_img.save(img_path)
        pred = None
        
        # Check if ResNet is available
        try:
            model = _get_resnet()
        except Exception as e:
            print(f"[ResNet] Model not available: {e}")
            model = None
        
        if model is not None:
            try:
                # Debug: check raw probabilities before thresholding
                with Image.open(img_path) as im:
                    img = im.convert("RGB")
                
                from resnet_classifier import _tfm, _DEVICE, _CLASSES
                import torch.nn.functional as F
                
                x = _tfm(256)(img).unsqueeze(0).to(_DEVICE)
                
                # 2-view TTA
                logits1 = model(x)
                logits2 = model(torch.flip(x, dims=[3]))
                logits = (logits1 + logits2) / 2
                
                probs = F.softmax(logits / 0.78, dim=1).squeeze(0)
                pvals, idxs = torch.sort(probs, descending=True)
                
                print(f"[ResNet] Debug - Top 3 raw probabilities:")
                for j in range(min(3, len(_CLASSES))):
                    class_name = _CLASSES[idxs[j]]
                    prob = float(pvals[j])
                    print(f"  {j+1}. {class_name}: {prob:.3f}")
                
                pred = diagnose_or_none(model, img_path, img_size=256)
                print(f"[ResNet] Raw prediction: {pred}")
            except FileNotFoundError as e:
                print(f"[ResNet] Model file not found: {e}")
                pred = None
            except torch.cuda.OutOfMemoryError as e:
                print(f"[ResNet] GPU memory error: {e}")
                pred = None
            except Exception as e:
                print(f"[ResNet] Error during prediction: {e}")
                import traceback
                traceback.print_exc()
                pred = None  # Ensure pred is None on error

        # Accept/canonize label once; no extra thresholds here
        final_label = "unknown"
        try:
            final_label = _accept_label(pred)  # returns e.g., {"healthy","frost injury","root rot","overwatering","drought","unknown"}
            print(f"[ResNet] Final label: {final_label}")
        except Exception as e:
            print(f"[ResNet] Error in _accept_label: {e}")
            pass

        # Confidence (optional badge)
        try:
            p1 = float(pred.get("p1", 0.0)) if pred else 0.0
        except Exception:
            p1 = 0.0

        # ---------------------------
        # system prompts
        # ---------------------------
        # Check if this looks like a non-plant image (person, object, etc.)
        # If confidence is very low or it's clearly not a plant, force unknown
        if final_label != "unknown" and pred:
            p1 = float(pred.get("p1", 0.0))
            # If confidence is below 65%, it's likely not a plant at all
            if p1 < 0.65:
                final_label = "unknown"
                print(f"[ResNet] Low confidence ({p1:.3f}) - treating as unknown")
            # Also check if it's predicting "healthy" with low confidence (likely non-plant)
            elif final_label.lower() == "healthy" and p1 < 0.75:
                final_label = "unknown"
                print(f"[ResNet] 'Healthy' with low confidence ({p1:.3f}) - likely non-plant, treating as unknown")
        
        if final_label == "unknown":
            # Softer prompt when we don't trust the classifier
            chat_state.system = """
<<SYS>>You are a plant diagnostician. Confidence is too low to choose a diagnosis.

RESPONSE STRUCTURE (follow exactly):
1. State: "Based on the image provided, I cannot confidently diagnose the plant with a specific disease or condition."

2. List exactly 3 possible causes:
   - Lack of close-up images: To accurately diagnose a plant, it's essential to examine its leaves, stems, and roots closely. Without close-up images of these areas, it's challenging to identify any issues or abnormalities.
   - Limited view of the plant: The image provided only shows the top portion of the plant, making it difficult to evaluate the overall health of the plant.
   - The image doesn't reveal any obvious signs of pests or diseases, such as holes in the leaves, discoloration, or unusual growths. Without more information, it's challenging to identify the cause of any issues.

3. List exactly 3 recommended fixes:
   - Close-up images of the leaves, stems, and roots to examine for any signs of pests, diseases, or abnormalities.
   - Images of the plant's overall growth, including its size, shape, and any unusual features.
   - Images of the soil and surrounding environment to assess the plant's root health and potential stressors.

4. End with: "Based on these additional images, we can begin to identify potential causes for the plant's decline or health. If you have any further questions or concerns, please feel free to ask."

FORMATTING REQUIREMENTS:
- Use compact numbered lists: "1. explanation" (not "1.\nexplanation")
- No extra line breaks between list items
- Keep explanations on the same line as numbers
- Use dashes (-) for bullet points, not asterisks (*)
- Follow the exact structure above - do not deviate

Do not provide differential possibilities or lengthy explanations.<</SYS>>
""".strip()
        else:
            # ── RF-DETR grounding ─────────────────────────────────────────
            detected_parts, all_detections = {}, []
            if RFDETR_AVAILABLE:
                detected_parts, all_detections = _run_rfdetr_on_image(img_path)
                if detected_parts:
                    vis = sorted(detected_parts.keys())
                    hid = sorted(PLANT_PARTS - set(vis))
                    print(f"[RF-DETR] Detected: {', '.join(vis)} | Not visible: {', '.join(hid)}")

            if detected_parts:
                visible_csv = ", ".join(sorted(detected_parts.keys()))
                hidden_csv = ", ".join(sorted(PLANT_PARTS - set(detected_parts.keys())))
                chat_state.system = f"""
<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {final_label.title()}
A vision-based detector has confirmed which plant parts are visible in this image.
VISIBLE (describe these): {visible_csv}.
NOT VISIBLE (do NOT mention these at all): {hidden_csv}.
You must ONLY describe symptoms on the VISIBLE parts listed above.
Do NOT name, reference, or explain anything involving {hidden_csv} — not even as causes, locations, or in advice.
The green sepals on top of a strawberry are the calyx, part of the fruit, not leaves.
Provide a detailed medical report:
1) Diagnosis: {final_label.title()}
2) Visible cues: Describe the visual symptoms on the detected parts.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough.<</SYS>>
""".strip()
            else:
                chat_state.system = f"""
<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {final_label.title()}
Your task is to examine the image and explain why this diagnosis is correct.
You MUST use this exact diagnosis: {final_label.title()}
Provide a detailed medical report in this format:
1) Diagnosis: {final_label.title()}
2) Visible cues: Describe the visual symptoms you observe that support this diagnosis.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough. Complete all recommendations fully.
<</SYS>>
""".strip()

        if user_message and user_message.strip():
            ask_text = user_message.strip()

            if RAG_AVAILABLE and disease_qa and final_label != "unknown":
                qa_result = disease_qa.answer_question(final_label, user_message)
                if qa_result.get("confidence", 0) >= 0.5:
                    qa_answer = qa_result.get("answer", "")
                    q_type = qa_result.get("question_type", "")
                    print(f"[QA-RAG] Detected {q_type} question, injecting targeted context")
                    ask_text += f"\n\nRelevant knowledge: {qa_answer}"
        else:
            if detected_parts:
                visible_csv = ", ".join(sorted(detected_parts.keys()))
                hidden_csv = ", ".join(sorted(PLANT_PARTS - set(detected_parts.keys())))
                ask_text = (
                    f"Analyze this strawberry plant image. The diagnosis is {final_label}. "
                    f"VISIBLE parts: {visible_csv}. NOT VISIBLE (do not mention): {hidden_csv}. "
                    f"Describe symptoms ONLY on the visible parts."
                )
            else:
                ask_text = f"Examine this image. The diagnosis is {final_label.title()}. Describe the visible symptoms and provide treatment recommendations."
        
        # NOTE: RAG and SERP are NOT injected into initial diagnosis prompt
        # - RAG is used for follow-up Q&A only (handled above and in is_followup check)
        # - SERP will be shown as supplementary info AFTER the model's response
        
        # Generation with hallucination check and regeneration
        max_attempts = 2  # Try up to 2 times if hallucination detected
        body = None
        hallucination_warning = None
        
        for attempt in range(max_attempts):
            # Reset conversation for each attempt
            if attempt > 0:
                chat_state = CONV_VISION.copy()
                img_list = []
                chat.upload_img(gr_img, chat_state, img_list)
                chat.encode_img(img_list)
                # Rebuild system prompt
                chat_state.system = f"""
<<SYS>>You are a plant diagnostician. The diagnosis has already been determined: {final_label.title()}
Your task is to examine the image and explain why this diagnosis is correct.
You MUST use this exact diagnosis: {final_label.title()}
Provide a detailed medical report in this format:
1) Diagnosis: {final_label.title()}
2) Visible cues: Describe the visual symptoms you observe that support this diagnosis.
3) Recommendation: Provide specific, actionable treatment steps.
Be detailed and thorough. Complete all recommendations fully.
IMPORTANT: Only describe symptoms of {final_label.title()}. Do not mention symptoms of other diseases.
<</SYS>>
""".strip()
                print(f"[Hallucination] Regenerating response (attempt {attempt + 1})")
            
            _ = chat.ask(ask_text, chat_state)

            ans = chat.answer(
                conv=chat_state,
                img_list=img_list,
                temperature=temperature,
                max_new_tokens=2000,
                max_length=4000,
                num_beams=1,
                repetition_penalty=1.01,
            )
            body = (ans[0] if isinstance(ans, (list, tuple)) and len(ans) else str(ans)).strip()
            body = _postprocess_caption(body)
            
            # Check for definite hallucinations (only for initial diagnosis)
            if RAG_AVAILABLE and disease_rag and final_label != "unknown" and not user_message.strip():
                hall_check = disease_rag.check_definite_hallucination(final_label, body)
                
                if hall_check.get("has_definite_hallucination"):
                    hallucinations = hall_check.get("hallucinations", [])
                    confused_with = hall_check.get("disease_confused_with", [])
                    print(f"[Hallucination] Detected: {hallucinations}")
                    
                    if attempt < max_attempts - 1:
                        # Will retry
                        continue
                    else:
                        # Final attempt still has hallucination - warn user
                        hallucination_warning = f"⚠️ *Note: Response may contain symptoms of {', '.join(confused_with)}. Please verify.*"
                        break
                else:
                    # No hallucination, accept this response
                    break
            else:
                # Not checking hallucinations for this case
                break
        
        # Optional confidence badge prefix
        if p1 > 0:
            badge = "🟢" if p1 >= 0.90 else "🟡" if p1 >= 0.70 else "🔴"
            body = f"{badge} **Confidence: {p1:.1%}**\n\n{body}"
        
        # Add hallucination warning if detected and couldn't fix
        if hallucination_warning:
            body += f"\n\n{hallucination_warning}"

        # Add SERP as supplementary info AFTER the model's response (not injected into prompt)
        # This provides additional context without overriding the trained detailed descriptions
        if is_enhanced and SERPAPI_AVAILABLE and final_label != "unknown" and not user_message.strip():
            try:
                serp_context = fetch_serp_context(f"strawberry {final_label} organic treatment management")
                if serp_context and len(serp_context) > 50:
                    body += "\n\n---\n**🌐 Web Resources:**\n"
                    body += f"_{serp_context[:500]}_"
            except Exception as e:
                print(f"[SERP] Error fetching context: {e}")

        # Add suggested follow-up questions (only for non-healthy initial diagnosis)
        if RAG_AVAILABLE and disease_qa and final_label not in ("unknown", "healthy") and not user_message.strip():
            suggestions = disease_qa.suggest_questions(final_label)[:3]
            if suggestions:
                body += "\n\n---\n**You can ask:**\n"
                for s in suggestions:
                    body += f"- {s}\n"

        # Build bbox image if toggled on
        bbox_img = None
        if show_bboxes and all_detections and gr_img is not None:
            try:
                pil_img = gr_img if isinstance(gr_img, Image.Image) else Image.fromarray(gr_img)
                bbox_img = _draw_bboxes(pil_img, all_detections)
            except Exception as e:
                print(f"[RF-DETR] Bbox drawing failed: {e}")

        # Clean up temporary file
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception:
            pass

        return (chatbot + [[user_message, body]], chat_state, img_list, bbox_img)

    except FileNotFoundError as e:
        error_msg = "❌ **Model Error**: Required model files not found. Please check that all model files are properly installed."
        logging.error(f"File not found: {str(e)}")
        return (chatbot + [[user_message, error_msg]], chat_state, img_list, None)
    except torch.cuda.OutOfMemoryError as e:
        error_msg = "❌ **Memory Error**: GPU memory insufficient. Please try with a smaller image or restart the application."
        logging.error(f"CUDA OOM: {str(e)}")
        return (chatbot + [[user_message, error_msg]], chat_state, img_list, None)
    except Exception as e:
        error_msg = "❌ **Unexpected Error**: An error occurred during processing. Please try again or contact support if the issue persists."
        logging.error(f"Unexpected error in chat processing: {str(e)}")
        return (chatbot + [[user_message, error_msg]], chat_state, img_list, None)



def reset_chat(chat_state, img_list):
    """Reset chat state and image list."""
    return [], CONV_VISION.copy(), []

# Load custom CSS from external file
def load_custom_css():
    """Load the dark theme CSS from external file."""
    # Try multiple possible CSS file locations
    css_paths = [
        Path(__file__).resolve().parent / "dark_theme.css",
        Path("dark_theme.css"),
        Path(__file__).parent / "dark_theme.css"
    ]
    
    # Enhanced fallback CSS
    fallback_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e0e0e0 !important;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        background: rgba(72, 187, 120, 0.8);
        color: white;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .custom-tab {
        background: rgba(20, 20, 35, 0.7);
        border-radius: 8px;
        margin: 5px;
    }
    .image-upload {
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 20px;
    }
    """
    
    for css_path in css_paths:
        try:
            if css_path.exists():
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                    print(f"[CSS] Loaded from: {css_path}")
                    return css_content
        except Exception as e:
            print(f"[CSS] Failed to load from {css_path}: {e}")
            continue
    
    print("[CSS] Using fallback CSS - no external CSS file found")
    return fallback_css

# Load the CSS
custom_css = load_custom_css()

# Import UI components
from ui_components import ABOUT_SECTION

# Create the Gradio interface with dark theme
with gr.Blocks(
    title="Plant Diagnostic System",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.neutral,
    ).set(
        # Background colors
        body_background_fill="#0d0d0d",
        body_background_fill_dark="#0d0d0d",
        background_fill_primary="#0d0d0d",
        background_fill_primary_dark="#0d0d0d",
        background_fill_secondary="#161616",
        background_fill_secondary_dark="#161616",
        # Block colors
        block_background_fill="#161616",
        block_background_fill_dark="#161616",
        block_border_color="#333333",
        block_border_color_dark="#333333",
        # Input colors
        input_background_fill="#1f1f1f",
        input_background_fill_dark="#1f1f1f",
        input_border_color="#333333",
        input_border_color_dark="#333333",
        # Text colors
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        body_text_color_subdued="#888888",
        body_text_color_subdued_dark="#888888",
        # Button colors
        button_primary_background_fill="#10a37f",
        button_primary_background_fill_dark="#10a37f",
        button_primary_background_fill_hover="#0d8a6a",
        button_primary_background_fill_hover_dark="#0d8a6a",
        button_secondary_background_fill="#1f1f1f",
        button_secondary_background_fill_dark="#1f1f1f",
    ),
    css=custom_css
) as demo:
    
    # State
    chat_state = gr.State(CONV_VISION.copy())
    img_list = gr.State([])
    original_img = gr.State(None)
    bbox_img_state = gr.State(None)
    
    # Header
    gr.Markdown("# Plant Diagnostic System")
    gr.Markdown("*AI-powered plant health analysis*")
    
    # Main Tabs
    with gr.Tabs():
        # === CHAT TAB ===
        with gr.TabItem("Chat"):
            with gr.Row(equal_height=True):
                # Left panel - controls
                with gr.Column(scale=1, min_width=280):
                    image = gr.Image(
                        type="pil", 
                        sources=["upload", "webcam"],
                        label="Upload Plant Image",
                        height=260
                    )
                    
                    new_chat_btn = gr.Button("New Analysis", elem_id="new-analysis-btn")
                    
                    with gr.Accordion("Settings", open=False, elem_id="settings-accordion"):
                        show_bboxes = gr.Checkbox(label="Show part detections", value=False, info="Display RF-DETR bounding boxes on uploaded image")
                        enhanced_mode = gr.Checkbox(label="Show web resources", value=False, info="Adds web search results after diagnosis")
                        temperature = gr.Slider(
                            minimum=0.01, maximum=0.5, value=0.2, step=0.01,
                            label="Temperature"
                        )
                    use_rag = gr.State(False)
                
                # Right panel - chat (fills remaining space)
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=600,
                        show_label=False,
                        elem_id="main-chatbot"
                    )
                    
                    # Input with integrated send button
                    with gr.Row(elem_id="chat-input-row"):
                        user_input = gr.Textbox(
                            placeholder="Describe the plant issue or ask a question...",
                            show_label=False,
                            container=False,
                            elem_id="chat-input",
                            scale=10
                        )
                        send_btn = gr.Button("➤", elem_id="send-btn", scale=1, min_width=50)
        
        # === KNOWLEDGE GRAPH TAB ===
        with gr.TabItem("Knowledge Graph"):
            with gr.Column(elem_id="kg-tab-container"):
                graph_plot = gr.Plot(label=None, elem_id="knowledge-graph-plot")
                
                with gr.Row(elem_id="kg-controls"):
                    disease_input = gr.Textbox(
                        label="",
                        placeholder="Enter disease (drought, gray_mold, root_rot...)",
                        container=False,
                        elem_id="disease-input",
                        scale=3
                    )
                    show_btn = gr.Button("View Disease", elem_id="kg-view-btn")
                    reload_graph = gr.Button("Show All", elem_id="kg-reset-btn")
                    gray_mold_btn = gr.Button("🦠 Gray Mold", elem_id="kg-gray-btn")
        
        # === ABOUT TAB ===
        with gr.TabItem("About"):
            gr.HTML(ABOUT_SECTION)
    
    # Chat handler
    def handle_chat(msg, history, state, img, imgs, temp, enhanced, rag_enabled, bboxes_on, orig_store, bbox_store):
        if not msg.strip() and img is None:
            return history or [], state, imgs, img, orig_store, bbox_store
        
        result = process_chat_with_image(
            msg, 
            history if history else [], 
            state, img, imgs, temp, 
            is_enhanced=enhanced,
            use_rag=rag_enabled,
            show_bboxes=True,
        )
        chatbot_out, state_out, imgs_out, bbox_img = result
        new_orig = img
        new_bbox = bbox_img
        display = bbox_img if (bboxes_on and bbox_img is not None) else img
        return chatbot_out, state_out, imgs_out, display, new_orig, new_bbox
    
    # Toggle handler — swap image in the same box
    def toggle_bboxes(bboxes_on, orig_store, bbox_store):
        if bboxes_on and bbox_store is not None:
            return bbox_store
        elif orig_store is not None:
            return orig_store
        return gr.update()
    
    # Reset handler
    def do_reset():
        return [], CONV_VISION.copy(), [], None, None, None

    # Event handlers
    chat_inputs = [user_input, chatbot, chat_state, image, img_list, temperature, enhanced_mode, use_rag, show_bboxes, original_img, bbox_img_state]
    chat_outputs = [chatbot, chat_state, img_list, image, original_img, bbox_img_state]
    
    send_btn.click(
        handle_chat, inputs=chat_inputs, outputs=chat_outputs
    ).then(lambda: "", None, user_input)
    
    user_input.submit(
        handle_chat, inputs=chat_inputs, outputs=chat_outputs
    ).then(lambda: "", None, user_input)
    
    show_bboxes.change(
        toggle_bboxes,
        inputs=[show_bboxes, original_img, bbox_img_state],
        outputs=[image],
    )
    
    new_chat_btn.click(
        do_reset,
        inputs=None,
        outputs=[chatbot, chat_state, img_list, image, original_img, bbox_img_state]
    )
    
    # Knowledge Graph events
    demo.load(fn=create_knowledge_graph, inputs=None, outputs=graph_plot)
    reload_graph.click(fn=create_knowledge_graph, inputs=None, outputs=graph_plot)
    show_btn.click(draw_disease_detail, inputs=[disease_input], outputs=[graph_plot])
    gray_mold_btn.click(lambda: "gray_mold", None, disease_input).then(draw_disease_detail, inputs=[disease_input], outputs=[graph_plot])

# Launch
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)

