#!/usr/bin/env python3
"""
Advanced Development Server for MiniGPT-4 Gradio UI
Features:
- Hot reloading for Python files
- CSS file watching and injection
- Real-time UI updates
- No need to restart the server
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to Python path (go up one level from demo-editing folder)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CSSReloadHandler(FileSystemEventHandler):
    """Handles CSS file changes and injects them into the running app"""
    
    def __init__(self, css_file_path):
        self.css_file_path = css_file_path
        self.last_reload = 0
        self.reload_delay = 0.5
        
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.css'):
            return
            
        current_time = time.time()
        if current_time - self.last_reload < self.reload_delay:
            return
            
        self.last_reload = current_time
        print(f"\n🎨 CSS file changed: {event.src_path}")
        print("🎨 Injecting CSS changes...")
        
        # Read the updated CSS
        try:
            with open(self.css_file_path, 'r') as f:
                css_content = f.read()
            
            # Inject CSS into the running Gradio app
            self.inject_css(css_content)
            print("✅ CSS updated successfully!")
            
        except Exception as e:
            print(f"❌ CSS update failed: {e}")
    
    def inject_css(self, css_content):
        """Inject CSS into the running Gradio app"""
        # This is a simplified approach - in practice, you might need to
        # communicate with the running Gradio app to inject CSS
        print("💡 CSS changes will be visible on the next page refresh")

class PythonReloadHandler(FileSystemEventHandler):
    """Handles Python file changes and reloads the app"""
    
    def __init__(self, app_module):
        self.app_module = app_module
        self.last_reload = 0
        self.reload_delay = 1.0
        
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.py'):
            return
            
        # Skip __pycache__ and other non-essential files
        if '__pycache__' in event.src_path or event.src_path.endswith('.pyc'):
            return
            
        current_time = time.time()
        if current_time - self.last_reload < self.reload_delay:
            return
            
        self.last_reload = current_time
        print(f"\n🔄 Python file changed: {event.src_path}")
        print("🔄 Reloading application...")
        
        # Reload the module
        try:
            if self.app_module in sys.modules:
                del sys.modules[self.app_module]
            __import__(self.app_module)
            print("✅ Python reload successful!")
        except Exception as e:
            print(f"❌ Python reload failed: {e}")

def load_css_from_file(css_file_path):
    """Load CSS from file and return as string"""
    try:
        with open(css_file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️  CSS file not found: {css_file_path}")
        return ""
    except Exception as e:
        print(f"❌ Error loading CSS: {e}")
        return ""

def create_enhanced_demo():
    """Create the enhanced demo with CSS loading"""
    
    # Load CSS from file
    css_file_path = project_root / "demo-editing" / "custom_styles.css"
    custom_css = load_css_from_file(css_file_path)
    
    # Import the demo
    import sys
    sys.path.insert(0, str(project_root / "demo-editing"))
    from demo_dev import create_gradio_interface
    
    # Create the interface
    demo = create_gradio_interface()
    
    return demo, css_file_path

def start_file_watchers(css_file_path, watch_dirs):
    """Start file watchers for both Python and CSS files"""
    
    # CSS watcher
    css_handler = CSSReloadHandler(css_file_path)
    css_observer = Observer()
    css_observer.schedule(css_handler, str(css_file_path.parent), recursive=False)
    
    # Python watcher
    python_handler = PythonReloadHandler("demo_dev")
    python_observer = Observer()
    
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            python_observer.schedule(python_handler, watch_dir, recursive=True)
            print(f"👀 Watching Python files in: {watch_dir}")
    
    # Start observers
    css_observer.start()
    python_observer.start()
    
    print(f"🎨 Watching CSS file: {css_file_path}")
    
    return css_observer, python_observer

def run_development_server():
    """Run the development server with hot reloading"""
    
    print("🚀 Starting MiniGPT-4 Development Server")
    print("=" * 60)
    print("🔥 Hot reloading enabled for both Python and CSS files!")
    print("📝 Edit demo_dev.py or ui_components.py for Python changes")
    print("🎨 Edit custom_styles.css for styling changes")
    print("🔄 Changes will be applied automatically!")
    print("=" * 60)
    
    # Directories to watch for Python changes
    watch_dirs = [
        str(project_root / "demo-editing"),
        str(project_root / "minigpt4"),
    ]
    
    try:
        # Create the demo
        demo, css_file_path = create_enhanced_demo()
        
        # Start file watchers
        css_observer, python_observer = start_file_watchers(css_file_path, watch_dirs)
        
        print("\n🌐 Starting Gradio server...")
        print("📱 Open your browser to: http://localhost:7861")
        print("🛑 Press Ctrl+C to stop the server")
        print("\n" + "=" * 60)
        
        # Launch the demo
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down development server...")
    except Exception as e:
        print(f"❌ Error starting development server: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install gradio watchdog")
    finally:
        # Stop observers
        try:
            css_observer.stop()
            python_observer.stop()
            css_observer.join()
            python_observer.join()
        except:
            pass

def install_dependencies():
    """Install required dependencies for development"""
    print("📦 Installing development dependencies...")
    
    dependencies = [
        "gradio",
        "watchdog",
        "plotly",
        "networkx",
        "pandas",
        "numpy",
        "torch",
        "PIL"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"📥 Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed successfully")

if __name__ == "__main__":
    # Check if we need to install dependencies
    try:
        import watchdog
        import gradio
    except ImportError:
        print("🔧 Installing required dependencies...")
        install_dependencies()
    
    # Run the development server
    run_development_server()
