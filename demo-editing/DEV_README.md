# 🌿 MiniGPT-4 UI Development Guide

This guide will help you develop the Gradio UI with real-time editing capabilities, eliminating the need to constantly launch and close the application.

## 🚀 Quick Start

### Option 1: Simple Start (Recommended)
```bash
# From the MiniGPT-4 project root
python demo-editing/start_dev.py
```

### Option 2: Advanced Development Server
```bash
# From the MiniGPT-4 project root
python demo-editing/dev_server_advanced.py
```

### Option 3: Basic Development Server
```bash
# From the MiniGPT-4 project root
python demo-editing/dev_server.py
```

## 📁 Development Files Structure

```
MiniGPT-4/
├── demo-editing/             # Development folder
│   ├── start_dev.py          # Quick start script
│   ├── dev_server_advanced.py # Advanced dev server with CSS watching
│   ├── dev_server.py         # Basic dev server
│   ├── demo_dev.py           # Development version of the demo
│   ├── ui_components.py      # Modular UI components
│   ├── custom_styles.css     # Custom CSS for styling
│   ├── example_ui_edit.py    # UI editing examples
│   ├── __init__.py           # Package initialization
│   └── DEV_README.md         # This file
└── [other project files...]
```

## 🎨 Real-Time Editing

### Python UI Changes
- Edit `demo-editing/ui_components.py` to modify UI components
- Edit `demo-editing/demo_dev.py` to change the main application logic
- Changes are automatically detected and the app reloads

### CSS Styling Changes
- Edit `demo-editing/custom_styles.css` to modify the visual appearance
- Changes are detected and injected in real-time
- No need to restart the server

## 🔧 Development Features

### Hot Reloading
- **Python files**: Automatically reloads when you save changes
- **CSS files**: Injects changes without restarting
- **No more launch-close cycles!**

### Modular Components
- **Header**: `create_header()` in `demo-editing/ui_components.py`
- **Image Analysis Tab**: `create_image_analysis_tab()`
- **Knowledge Graph Tab**: `create_knowledge_graph_tab()`
- **Settings Tab**: `create_settings_tab()`

### Custom Styling
- All CSS is in `demo-editing/custom_styles.css`
- Dark theme with cyan accents
- Responsive design
- Smooth animations and transitions

## 🎯 UI Development Workflow

1. **Start the development server**:
   ```bash
   python demo-editing/start_dev.py
   ```

2. **Open your browser** to `http://localhost:7860`

3. **Edit UI components** in `demo-editing/ui_components.py`:
   - Modify component layouts
   - Add new UI elements
   - Change component properties

4. **Edit styling** in `demo-editing/custom_styles.css`:
   - Change colors, fonts, spacing
   - Add animations
   - Modify responsive behavior

5. **See changes instantly** without restarting!

## 📝 Example: Adding a New UI Component

### 1. Add to `ui_components.py`:
```python
def create_new_component():
    with gr.Group():
        gr.Markdown("### New Component")
        new_input = gr.Textbox(label="New Input")
        new_button = gr.Button("New Action")
    
    return {
        'new_input': new_input,
        'new_button': new_button
    }
```

### 2. Use in `demo_dev.py`:
```python
# In create_gradio_interface()
new_tab, new_components = create_new_component()

# Add event handlers
new_components['new_button'].click(
    some_function,
    inputs=[new_components['new_input']],
    outputs=[some_output]
)
```

### 3. Style in `custom_styles.css`:
```css
.new-component {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 8px;
    padding: 15px;
}
```

## 🎨 CSS Development Tips

### Color Scheme
- **Primary**: `#00d4ff` (Cyan)
- **Background**: `rgba(20, 20, 35, 0.8)` (Dark blue)
- **Text**: `#e0e0e0` (Light gray)
- **Accent**: `rgba(0, 212, 255, 0.3)` (Cyan with opacity)

### Common Classes
- `.status-badge`: Status indicators
- `.custom-tab`: Tab containers
- `.chatbot-custom`: Chat interfaces
- `.text-input-custom`: Text inputs
- `.graph-container`: Graph visualizations

### Responsive Design
```css
@media (max-width: 768px) {
    /* Mobile styles */
}
```

## 🐛 Troubleshooting

### Server won't start
```bash
# Install dependencies
pip install gradio watchdog plotly networkx pandas numpy torch PIL

# Check file permissions
chmod +x start_dev.py dev_server_advanced.py demo_dev.py
```

### Changes not appearing
- Check that files are being saved
- Look for error messages in the terminal
- Try refreshing the browser page
- Check that the file watcher is running

### CSS not updating
- Make sure `custom_styles.css` exists
- Check for CSS syntax errors
- Try refreshing the browser

## 🚀 Production Deployment

When you're ready to deploy:

1. **Test your changes** thoroughly
2. **Copy your UI components** to the main demo file
3. **Update the main CSS** in the production file
4. **Test the production version**

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [CSS Reference](https://developer.mozilla.org/en-US/docs/Web/CSS)
- [Python Watchdog](https://python-watchdog.readthedocs.io/)

## 💡 Tips for Better Development

1. **Use the browser's developer tools** to inspect elements
2. **Test on different screen sizes** using responsive design
3. **Keep components modular** for easier maintenance
4. **Use meaningful class names** in CSS
5. **Test frequently** as you make changes

Happy coding! 🎉
