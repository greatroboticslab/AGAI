#!/usr/bin/env python3
"""
Example: How to edit the UI in real-time
This shows you how to make changes to the UI components
"""

# Example 1: Adding a new status badge to the header
def create_header_with_new_badge():
    """Example of modifying the header component"""
    return """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3em; margin-bottom: 10px;">
                🌿 Plant Diagnostic System
            </h1>
            <p style="color: #a0a0a0; font-size: 1.2em;">
                Advanced AI-Powered Strawberry Plant Health Analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                <span class="status-badge status-healthy">✓ System Online</span>
                <span class="status-badge" style="background: rgba(102, 126, 234, 0.8); color: white;">
                    🔬 ResNet + MiniGPT-v2
                </span>
                <span class="status-badge" style="background: rgba(183, 148, 244, 0.8); color: white;">
                    📊 Knowledge Graph Ready
                </span>
                <!-- NEW BADGE ADDED HERE -->
                <span class="status-badge" style="background: rgba(34, 197, 94, 0.8); color: white;">
                    🚀 Dev Mode Active
                </span>
            </div>
        </div>
    """

# Example 2: Adding a new input field to the image analysis tab
def create_enhanced_image_analysis():
    """Example of adding new components to existing tabs"""
    
    # This would go in ui_components.py in the create_image_analysis_tab function
    enhanced_controls = """
        # Add this after the temperature slider in create_image_analysis_tab()
        
        # NEW: Analysis mode selector
        analysis_mode = gr.Radio(
            choices=["Quick Analysis", "Detailed Analysis", "Expert Analysis"],
            value="Quick Analysis",
            label="🔍 Analysis Mode",
            info="Choose the depth of analysis"
        )
        
        # NEW: Confidence threshold
        confidence_threshold = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.7,
            step=0.05,
            label="🎯 Confidence Threshold",
            info="Minimum confidence for diagnosis"
        )
        
        # NEW: Save results checkbox
        save_results = gr.Checkbox(
            value=False,
            label="💾 Save Analysis Results",
            info="Automatically save results to file"
        )
    """
    
    return enhanced_controls

# Example 3: Adding custom CSS for new components
def create_custom_css_examples():
    """Example CSS additions for new components"""
    
    css_examples = """
    /* Add these styles to custom_styles.css */
    
    /* NEW: Analysis mode radio buttons */
    .analysis-mode-radio {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 10px 0 !important;
    }
    
    /* NEW: Confidence threshold slider */
    .confidence-slider {
        background: linear-gradient(90deg, 
            rgba(255, 107, 107, 0.3) 0%, 
            rgba(255, 193, 7, 0.3) 50%, 
            rgba(34, 197, 94, 0.3) 100%) !important;
    }
    
    /* NEW: Save results checkbox */
    .save-results-checkbox {
        background: rgba(0, 212, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.1) !important;
        border-radius: 6px !important;
        padding: 8px !important;
    }
    
    /* NEW: Custom button styles */
    .btn-analyze {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-analyze:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* NEW: Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: #22c55e; }
    .status-warning { background: #f59e0b; }
    .status-error { background: #ef4444; }
    """
    
    return css_examples

# Example 4: How to add event handlers for new components
def create_event_handler_examples():
    """Example of adding event handlers for new components"""
    
    event_examples = """
    # Add these to demo_dev.py in the create_gradio_interface() function
    
    # NEW: Analysis mode change handler
    def on_analysis_mode_change(mode):
        if mode == "Expert Analysis":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    analysis_components['analysis_mode'].change(
        on_analysis_mode_change,
        inputs=[analysis_components['analysis_mode']],
        outputs=[
            analysis_components['confidence_threshold'],
            analysis_components['save_results']
        ]
    )
    
    # NEW: Enhanced analysis with mode and confidence
    def enhanced_analysis_with_mode(input_text, chatbot, chat_state, image, img_list, 
                                  temperature, analysis_mode, confidence_threshold):
        # Your analysis logic here
        response = f"Analysis mode: {analysis_mode}, Confidence: {confidence_threshold}"
        chatbot.append([input_text, response])
        chat_state.append([input_text, response])
        return chatbot, chat_state, img_list
    
    analysis_components['enhanced_send'].click(
        enhanced_analysis_with_mode,
        inputs=[
            analysis_components['enhanced_input'],
            analysis_components['enhanced_chatbot'],
            analysis_components['enhanced_chat_state'],
            analysis_components['image'],
            analysis_components['enhanced_img_list'],
            analysis_components['temperature'],
            analysis_components['analysis_mode'],
            analysis_components['confidence_threshold']
        ],
        outputs=[
            analysis_components['enhanced_chatbot'],
            analysis_components['enhanced_chat_state'],
            analysis_components['enhanced_img_list']
        ]
    )
    """
    
    return event_examples

if __name__ == "__main__":
    print("📝 UI Development Examples")
    print("=" * 50)
    print("These are examples of how to modify the UI components.")
    print("Copy the relevant code to the appropriate files:")
    print()
    print("1. Header modifications → ui_components.py")
    print("2. New components → ui_components.py")
    print("3. CSS styles → custom_styles.css")
    print("4. Event handlers → demo_dev.py")
    print()
    print("Then run: python start_dev.py")
    print("Your changes will appear in real-time!")
