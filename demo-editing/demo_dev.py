#!/usr/bin/env python3
"""
Development version of MiniGPT-4 Plant Diagnostic System
This version is optimized for real-time UI development with hot reloading
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Disable analytics to reduce connection issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# UI-only mode flag
UI_ONLY_MODE = os.environ.get("UI_ONLY", "0") == "1"

# Add project root to path (go up one level from demo-editing folder)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "demo-editing"))

# Third-party imports
import gradio as gr
import torch

# MiniGPT-4 imports
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Import UI components
import sys
sys.path.insert(0, str(project_root / "demo-editing"))
from ui_components import (
    create_header,
    create_image_analysis_tab,
    create_knowledge_graph_tab,
    create_about_tab,
    create_custom_css,
    create_knowledge_graph
)

# Import model functions (you can mock these for UI development)
try:
    from resnet_classifier import load_resnet, diagnose_or_none
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  Model not available - running in UI-only mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model (will be loaded when needed)
model = None
vis_processor = None
chat = None

def load_model():
    """Load the MiniGPT-4 model (can be mocked for UI development)"""
    global model, vis_processor, chat
    
    if UI_ONLY_MODE:
        logger.info("🌐 UI_ONLY=1 - Skipping model loading for faster development")
        return
        
    if not MODEL_AVAILABLE:
        logger.info("Running in UI-only mode - model not loaded")
        return
    
    try:
        # Load model configuration
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg-path", default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--options", nargs="+", help="modify some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), modify by using -o")
        args = parser.parse_args([])  # Use empty args for now
        
        # Load config
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        
        # Load processor
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        # Initialize chat
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("Continuing in UI-only mode...")

def mock_process_chat_with_image(input_text, chatbot, chat_state, image, img_list, temperature=0.2):
    """Mock function for processing chat with image (for UI development)"""
    if image is None:
        return chatbot, chat_state, img_list
    
    # Mock response for UI development
    mock_responses = [
        "🔍 **Plant Analysis Complete**\n\nI can see this is a strawberry plant. Based on the image analysis:\n\n✅ **Overall Health**: Good\n🌱 **Growth Stage**: Fruiting\n🍃 **Leaf Condition**: Healthy green foliage\n\n**Recommendations**:\n- Continue regular watering\n- Monitor for any discoloration\n- Consider adding mulch around the base",
        "🔬 **Detailed Analysis**\n\n**Plant Type**: Fragaria × ananassa (Strawberry)\n**Health Status**: 85% healthy\n\n**Key Observations**:\n- Strong root system visible\n- Good leaf density\n- No visible signs of disease\n- Proper fruit development\n\n**Next Steps**:\n- Weekly inspection recommended\n- Fertilize in 2 weeks\n- Watch for pest activity",
        "⚠️ **Potential Issues Detected**\n\nI notice some concerning signs in your strawberry plant:\n\n🔴 **Issues Found**:\n- Slight yellowing on lower leaves\n- Possible nutrient deficiency\n- Minor pest damage on edges\n\n**Recommended Actions**:\n- Check soil pH (should be 6.0-6.5)\n- Apply balanced fertilizer\n- Inspect for aphids or mites\n- Improve air circulation"
    ]
    
    import random
    response = random.choice(mock_responses)
    
    # Update chat
    chatbot.append([input_text, response])
    chat_state.append([input_text, response])
    
    return chatbot, chat_state, img_list

def process_chat_with_image(input_text, chatbot, chat_state, image, img_list, temperature=0.2):
    """Process chat with image using the actual model or mock"""
    if model is None or not MODEL_AVAILABLE:
        return mock_process_chat_with_image(input_text, chatbot, chat_state, image, img_list, temperature)
    
    # Real model processing would go here
    # For now, use mock
    return mock_process_chat_with_image(input_text, chatbot, chat_state, image, img_list, temperature)

def reset_chat(chat_state, img_list):
    """Reset chat state and image list"""
    return [], [], [], []


def healthcheck():
    """Simple health check function"""
    return "OK"

def create_gradio_interface():
    """Create the main Gradio interface with dev-safe fallback"""
    
    # Load custom CSS
    custom_css = create_custom_css()
    
    # Create the main interface
    with gr.Blocks(
        title="🌿 Plant Diagnostic System",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            # Force dark backgrounds globally
            body_background_fill="linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%)",
            body_background_fill_dark="linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%)",
            background_fill_primary="rgba(20, 20, 35, 0.9)",
            background_fill_primary_dark="rgba(20, 20, 35, 0.9)",
            background_fill_secondary="rgba(25, 25, 40, 0.8)",
            background_fill_secondary_dark="rgba(25, 25, 40, 0.8)",
            border_color_primary="rgba(0, 212, 255, 0.2)",
            border_color_primary_dark="rgba(0, 212, 255, 0.2)",
            # Text colors
            body_text_color="#e0e0e0",
            body_text_color_dark="#e0e0e0",
            body_text_color_subdued="#a0a0a0",
            body_text_color_subdued_dark="#a0a0a0",
            # Component colors
            color_accent_soft="rgba(0, 212, 255, 0.1)",
            color_accent_soft_dark="rgba(0, 212, 255, 0.1)",
            # Block styling
            block_background_fill="rgba(20, 20, 35, 0.7)",
            block_background_fill_dark="rgba(20, 20, 35, 0.7)",
            block_border_color="rgba(0, 212, 255, 0.15)",
            block_border_color_dark="rgba(0, 212, 255, 0.15)",
            block_label_background_fill="transparent",
            block_label_background_fill_dark="transparent",
            # Input styling
            input_background_fill="rgba(10, 10, 20, 0.8)",
            input_background_fill_dark="rgba(10, 10, 20, 0.8)",
            input_border_color="rgba(0, 212, 255, 0.2)",
            input_border_color_dark="rgba(0, 212, 255, 0.2)",
            input_border_color_focus="rgba(0, 212, 255, 0.5)",
            input_border_color_focus_dark="rgba(0, 212, 255, 0.5)",
        ),
        css=custom_css
    ) as demo:
        try:
            # Header
            create_header()
            
            # Main tabs
            with gr.Tabs() as tabs:
                # Image Analysis Tab
                analysis_tab, analysis_components = create_image_analysis_tab()
                
                # Knowledge Graph Tab
                graph_tab, graph_components = create_knowledge_graph_tab()
                
                # About Tab
                about_tab, about_components = create_about_tab()
            
            # Event handlers for Image Analysis
            analysis_components['basic_send'].click(
                process_chat_with_image,
                inputs=[analysis_components['basic_input'], analysis_components['basic_chatbot'], analysis_components['basic_chat_state'], analysis_components['image'], analysis_components['basic_img_list'], analysis_components['temperature']],
                outputs=[analysis_components['basic_chatbot'], analysis_components['basic_chat_state'], analysis_components['basic_img_list']]
            ).then(lambda: "", None, analysis_components['basic_input'])
            
            analysis_components['basic_input'].submit(
                process_chat_with_image,
                inputs=[analysis_components['basic_input'], analysis_components['basic_chatbot'], analysis_components['basic_chat_state'], analysis_components['image'], analysis_components['basic_img_list'], analysis_components['temperature']],
                outputs=[analysis_components['basic_chatbot'], analysis_components['basic_chat_state'], analysis_components['basic_img_list']]
            ).then(lambda: "", None, analysis_components['basic_input'])
            
            analysis_components['enhanced_send'].click(
                lambda *args: process_chat_with_image(*args, is_enhanced=True),
                inputs=[analysis_components['enhanced_input'], analysis_components['enhanced_chatbot'], analysis_components['enhanced_chat_state'], analysis_components['image'], analysis_components['enhanced_img_list'], analysis_components['temperature']],
                outputs=[analysis_components['enhanced_chatbot'], analysis_components['enhanced_chat_state'], analysis_components['enhanced_img_list']]
            ).then(lambda: "", None, analysis_components['enhanced_input'])
            
            analysis_components['enhanced_input'].submit(
                lambda *args: process_chat_with_image(*args, is_enhanced=True),
                inputs=[analysis_components['enhanced_input'], analysis_components['enhanced_chatbot'], analysis_components['enhanced_chat_state'], analysis_components['image'], analysis_components['enhanced_img_list'], analysis_components['temperature']],
                outputs=[analysis_components['enhanced_chatbot'], analysis_components['enhanced_chat_state'], analysis_components['enhanced_img_list']]
            ).then(lambda: "", None, analysis_components['enhanced_input'])
            
            analysis_components['clear'].click(
                reset_chat,
                inputs=[analysis_components['basic_chat_state'], analysis_components['basic_img_list']],
                outputs=[analysis_components['basic_chatbot'], analysis_components['basic_chat_state'], analysis_components['basic_img_list']]
            ).then(
                reset_chat,
                inputs=[analysis_components['enhanced_chat_state'], analysis_components['enhanced_img_list']],
                outputs=[analysis_components['enhanced_chatbot'], analysis_components['enhanced_chat_state'], analysis_components['enhanced_img_list']]
            ).then(lambda: None, None, analysis_components['image'])
            
            # Knowledge Graph events
            demo.load(fn=create_knowledge_graph, inputs=None, outputs=graph_components['graph_plot'])
            graph_components['reload_graph'].click(fn=create_knowledge_graph, inputs=None, outputs=graph_components['graph_plot'])
            
            # Initialize graph safely
            try:
                graph_components['graph_plot'].value = create_knowledge_graph()
            except Exception as e:
                gr.Markdown(f"⚠️ Graph failed to render: `{e}`")

        except Exception as e:
            tb = traceback.format_exc()
            with gr.Column():
                gr.Markdown("## ❌ UI failed to build — dev-safe fallback")
                gr.Code(tb, language="python")
                gr.Markdown("Fix the error and save — hot reload will rebuild automatically.")
    
    return demo

def main():
    """Main function to run the development server"""
    print("🚀 Starting MiniGPT-4 Plant Diagnostic System - Development Mode")
    print("=" * 60)
    
    # Load model (optional for UI development)
    load_model()
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    print("🌐 Development server starting...")
    print("📝 Edit ui_components.py to modify the UI in real-time!")
    print("🎨 Edit the CSS in ui_components.py for styling changes!")
    print("🔄 The server will auto-reload when you make changes!")
    print("\n" + "=" * 60)
    
    # Health check
    print("🩺 healthcheck:", healthcheck())
    
    # Launch with dev-friendly settings
    try:
        demo.launch(
            server_name="0.0.0.0",   # so you can reach it from anywhere
            server_port=7861,        # stay on 7861 as requested
            show_error=True,
            quiet=False,
            inbrowser=False,
            favicon_path=None,
            ssl_verify=False
        )
    except OSError as e:
        print(f"❌ OSError during demo.launch(): {e}")
        print(f"❌ Full error details: {traceback.format_exc()}")
        print("💡 Try a different port or check if port 7861 is already in use")
        raise

if __name__ == "__main__":
    main()
