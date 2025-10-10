#!/usr/bin/env python3
"""
Minimal test server to verify Gradio works
"""

import os
import sys
from pathlib import Path

# Disable analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr

def healthcheck():
    return "OK"

def simple_interface():
    with gr.Blocks(title="Test Server") as demo:
        gr.Markdown("# 🌿 Test Server")
        gr.Textbox(value="Server is working!", label="Status")
        gr.Button("Test").click(healthcheck, outputs=gr.Textbox())
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting minimal test server...")
    demo = simple_interface()
    print("🩺 healthcheck:", healthcheck())
    print("📱 Server will be at: http://localhost:7861")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        enable_queue=False,
        show_error=True,
        quiet=False,
        inbrowser=False
    )
