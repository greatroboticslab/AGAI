"""
UI Components for MiniGPT-4 Plant Diagnostic System
Separated for easier development and real-time editing
"""

import gradio as gr
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
from datetime import datetime

def draw_crop_from_csv(*args, **kwargs):
    """Temporary stub so UI builds; replace with real logic later"""
    return create_knowledge_graph()

def create_header() -> gr.HTML:
    """Create the header section with animated gradient"""
    # 🔥 HOT RELOAD TEST - This print statement should appear when the server restarts
    stamp = datetime.now().strftime("%H:%M:%S")
    print("🔥🔥🔥 HOT RELOAD: create_header() function loaded at", stamp, "🔥🔥🔥")
    
    return gr.HTML(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3em; margin-bottom: 10px;">
    neon pink edition
            
            </h1>
            <p style="color: #a0a0a0; font-size: 1.2em;">
                Advanced AI-Powered Strawberry Plant Health Analysis
            </p>
            <div style="margin-top: 8px; font-size: 0.9em; color: #8aa;">
                Dev build: <code>{stamp}</code>
            </div>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                <span class="status-badge status-healthy">✓ System Online</span>
                <span class="status-badge" style="background: rgba(102, 126, 234, 0.8); color: white;">
                    🔬 ResNet + MiniGPT-v2
                </span>
                <span class="status-badge" style="background: rgba(183, 148, 244, 0.8); color: white;">
                    📊 Knowledge Graph Ready
                </span>
            </div>
        </div>
    """)

def create_image_analysis_tab() -> Tuple[gr.Tab, Dict[str, Any]]:
    """Create the image analysis tab with all its components"""
    
    with gr.Tab("🔬 Image Analysis", elem_classes="custom-tab") as tab:
        with gr.Row():
            # Left column - Image upload and controls
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Upload Image")
                image = gr.Image(
                    type="pil", 
                    label="Plant Image",
                    elem_classes="image-upload"
                )
                
                # Analysis settings card
                with gr.Group():
                    gr.Markdown("### ⚙️ Analysis Settings")
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=0.5,
                        value=0.2,
                        step=0.01,
                        label="🌡️ Temperature",
                        info="Lower = More focused | Higher = More creative"
                    )
                    
                    # Analysis status
                    gr.HTML("""
                        <div style="padding: 10px; background: rgba(0, 212, 255, 0.1); 
                                    border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.3);">
                            <p style="margin: 0; color: #00d4ff; font-size: 0.9em;">
                                💡 Tip: Upload a clear image of your strawberry plant for best results
                            </p>
                        </div>
                    """)
                    
                clear = gr.Button(
                    "🔄 Reset All", 
                    variant="stop",
                    size="lg"
                )

            # Right column - Chat interfaces
            with gr.Column(scale=2):
                # Standard Analysis
                with gr.Group():
                    gr.Markdown("### 🤖 Standard Analysis")
                    gr.Markdown("*Quick diagnosis based on visual inspection*")
                    
                    basic_chatbot = gr.Chatbot(
                        height=300,
                        bubble_full_width=False,
                        avatar_images=["🧑‍🌾", "🤖"]
                    )
                    basic_chat_state = gr.State()
                    basic_img_list = gr.State([])
                    
                    with gr.Row():
                        basic_input = gr.Textbox(
                            placeholder="Ask about the plant's condition...",
                            scale=4,
                            label=None,
                            container=False
                        )
                        basic_send = gr.Button(
                            "📤 Send", 
                            scale=1, 
                            variant="primary"
                        )

                # Enhanced Analysis
                with gr.Group():
                    gr.Markdown("### 🔍 Enhanced Analysis")
                    gr.Markdown("*Comprehensive diagnosis with web research*")
                    
                    enhanced_chatbot = gr.Chatbot(
                        height=300,
                        bubble_full_width=False,
                        avatar_images=["🧑‍🌾", "🔬"]
                    )
                    enhanced_chat_state = gr.State()
                    enhanced_img_list = gr.State([])
                    
                    with gr.Row():
                        enhanced_input = gr.Textbox(
                            placeholder="Get detailed analysis with treatment recommendations...",
                            scale=4,
                            label=None,
                            container=False
                        )
                        enhanced_send = gr.Button(
                            "🔎 Analyze", 
                            scale=1, 
                            variant="primary"
                        )

    # Return the tab and all components for event binding
    components = {
        'image': image,
        'temperature': temperature,
        'clear': clear,
        'basic_chatbot': basic_chatbot,
        'basic_chat_state': basic_chat_state,
        'basic_img_list': basic_img_list,
        'basic_input': basic_input,
        'basic_send': basic_send,
        'enhanced_chatbot': enhanced_chatbot,
        'enhanced_chat_state': enhanced_chat_state,
        'enhanced_img_list': enhanced_img_list,
        'enhanced_input': enhanced_input,
        'enhanced_send': enhanced_send,
    }
    
    return tab, components

def create_knowledge_graph_tab() -> Tuple[gr.Tab, Dict[str, Any]]:
    """Create the knowledge graph visualization tab"""
    
    with gr.Tab("📊 Knowledge Graph", elem_classes="custom-tab") as tab:
        gr.Markdown("""
            ### 🌐 FAOSTAT Agricultural Knowledge Network
            <span style="color: #a0a0a0;">Explore relationships between crops, conditions, and agricultural data</span>
        """)
        
        # Graph display with full width
        graph_plot = gr.Plot(label=None)
        
        # Controls below the graph
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎛️ Graph Controls")
                    reload_graph = gr.Button(
                        "🔄 Reload Full Graph",
                        variant="primary",
                        size="lg"
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎯 Crop Explorer")
                    crop_id = gr.Textbox(
                        label="Crop ID",
                        placeholder="e.g., 144 (strawberries)",
                        info="View specific crop neighborhood"
                    )
                    show_btn = gr.Button(
                        "🔍 Show Neighborhood",
                        variant="secondary",
                        size="lg"
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚡ Quick Access")
                    gr.Button("🍓 Strawberries", size="sm").click(
                        lambda: "144", None, crop_id
                    ).then(
                        draw_crop_from_csv, inputs=[crop_id], outputs=[graph_plot]
                    )
                    gr.Button("🍅 Tomatoes", size="sm").click(
                        lambda: "388", None, crop_id
                    ).then(
                        draw_crop_from_csv, inputs=[crop_id], outputs=[graph_plot]
                    )

    components = {
        'reload_graph': reload_graph,
        'graph_plot': graph_plot,
        'crop_id': crop_id,
        'show_btn': show_btn,
    }
    
    return tab, components

def create_about_tab() -> Tuple[gr.Tab, Dict[str, Any]]:
    """Create the about tab with system information"""
    
    with gr.Tab("ℹ️ About", elem_classes="custom-tab") as tab:
        gr.HTML("""
        <div style="padding: 20px; color: #e0e0e0;">
            <h2 style="color: #00d4ff; margin-bottom: 20px;">🌿 Plant Diagnostic System v2.0</h2>
            
            <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <h3 style="color: #b794f4; margin-bottom: 15px;">🎯 Features</h3>
                <ul style="color: #d0d0d0; line-height: 1.8;">
                    <li><strong style="color: #00d4ff;">Dual AI Analysis:</strong> ResNet classifier + MiniGPT-v2 vision model</li>
                    <li><strong style="color: #00d4ff;">Web Integration:</strong> Real-time information from SERPAPI</li>
                    <li><strong style="color: #00d4ff;">Knowledge Graph:</strong> Interactive FAOSTAT agricultural data visualization</li>
                    <li><strong style="color: #00d4ff;">Modern UI:</strong> Dark theme with responsive design</li>
                </ul>
            </div>
            
            <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <h3 style="color: #b794f4; margin-bottom: 15px;">🛠️ Technology Stack</h3>
                <ul style="color: #d0d0d0; line-height: 1.8;">
                    <li><strong style="color: #48bb78;">Vision Models:</strong> ResNet (classification) + MiniGPT-v2 (description)</li>
                    <li><strong style="color: #48bb78;">Data:</strong> FAOSTAT agricultural database</li>
                    <li><strong style="color: #48bb78;">Visualization:</strong> Plotly + NetworkX</li>
                    <li><strong style="color: #48bb78;">Framework:</strong> Gradio with custom theming</li>
                </ul>
            </div>
            
            <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <h3 style="color: #b794f4; margin-bottom: 15px;">📈 Confidence Indicators</h3>
                <div style="display: flex; flex-direction: column; gap: 10px; color: #d0d0d0;">
                    <div><span style="font-size: 1.2em;">🟢</span> <strong style="color: #48bb78;">High Confidence</strong> (>90%): Highly reliable diagnosis</div>
                    <div><span style="font-size: 1.2em;">🟡</span> <strong style="color: #ffd93d;">Medium Confidence</strong> (70-90%): Good reliability</div>
                    <div><span style="font-size: 1.2em;">🔴</span> <strong style="color: #ff6b6b;">Low Confidence</strong> (<70%): Further inspection recommended</div>
                </div>
            </div>
            
            <div style="background: rgba(25, 25, 40, 0.8); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <h3 style="color: #b794f4; margin-bottom: 15px;">🔍 Best Practices</h3>
                <ol style="color: #d0d0d0; line-height: 1.8;">
                    <li>Upload clear, well-lit images of affected plant areas</li>
                    <li>Include both close-ups and full plant views when possible</li>
                    <li>Use Enhanced Analysis for detailed treatment recommendations</li>
                    <li>Explore the Knowledge Graph for related agricultural insights</li>
                </ol>
            </div>
            
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(183, 148, 244, 0.1)); border-radius: 12px; border: 1px solid rgba(0, 212, 255, 0.3);">
                <p style="color: #00d4ff; margin: 0; font-size: 1.1em;">
                    ✅ System Status: All models loaded and operational
                </p>
            </div>
        </div>
        """)

    components = {}
    
    return tab, components

def create_custom_css() -> str:
    """Create custom CSS for the application"""
    # 🔥 HOT RELOAD TEST - This print statement should appear when CSS changes
    print("🎨 HOT RELOAD: CSS loaded at", __import__('datetime').datetime.now().strftime("%H:%M:%S"))
    
    return """
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

def create_knowledge_graph() -> go.Figure:
    """Create a sample knowledge graph for demonstration"""
    # Sample data for the knowledge graph
    nodes = [
        {"id": "Strawberry", "group": "plant", "size": 20},
        {"id": "Powdery Mildew", "group": "disease", "size": 15},
        {"id": "Gray Mold", "group": "disease", "size": 15},
        {"id": "Leaf Spot", "group": "disease", "size": 12},
        {"id": "White Spots", "group": "symptom", "size": 10},
        {"id": "Fuzzy Growth", "group": "symptom", "size": 10},
        {"id": "Brown Spots", "group": "symptom", "size": 10},
        {"id": "Fungicide", "group": "treatment", "size": 12},
        {"id": "Pruning", "group": "treatment", "size": 10},
        {"id": "Air Circulation", "group": "prevention", "size": 8},
    ]
    
    edges = [
        ("Strawberry", "Powdery Mildew"),
        ("Strawberry", "Gray Mold"),
        ("Strawberry", "Leaf Spot"),
        ("Powdery Mildew", "White Spots"),
        ("Gray Mold", "Fuzzy Growth"),
        ("Leaf Spot", "Brown Spots"),
        ("Powdery Mildew", "Fungicide"),
        ("Gray Mold", "Pruning"),
        ("Leaf Spot", "Fungicide"),
        ("Fungicide", "Air Circulation"),
    ]
    
    # Create the graph
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=2, color='rgba(0, 212, 255, 0.3)'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    # Simple layout (in a real app, you'd use a proper graph layout algorithm)
    positions = {
        "Strawberry": (0, 0),
        "Powdery Mildew": (-1, 1),
        "Gray Mold": (0, 1),
        "Leaf Spot": (1, 1),
        "White Spots": (-1.5, 2),
        "Fuzzy Growth": (0, 2),
        "Brown Spots": (1.5, 2),
        "Fungicide": (-0.5, 3),
        "Pruning": (0.5, 3),
        "Air Circulation": (0, 4),
    }
    
    color_map = {
        "plant": "#00d4ff",
        "disease": "#ff6b6b",
        "symptom": "#ffd93d",
        "treatment": "#6bcf7f",
        "prevention": "#a78bfa",
    }
    
    for node in nodes:
        x, y = positions[node["id"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node["id"])
        node_colors.append(color_map[node["group"]])
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=[node["size"] for node in nodes],
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        hoverinfo='text',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title="Plant Disease Knowledge Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive knowledge graph showing plant diseases, symptoms, and treatments",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="#a0a0a0", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(20, 20, 35, 0.8)',
        paper_bgcolor='rgba(20, 20, 35, 0.8)',
        font=dict(color='#e0e0e0')
    )
    
    return fig
