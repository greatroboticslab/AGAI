#!/usr/bin/env python3
"""
Development server for MiniGPT-4 Gradio UI with hot reloading
This allows you to edit the UI in real-time without restarting the server
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class GradioReloadHandler(FileSystemEventHandler):
    """Handles file changes and reloads the Gradio app"""
    
    def __init__(self, app_module):
        self.app_module = app_module
        self.last_reload = 0
        self.reload_delay = 1.0  # Minimum delay between reloads
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only reload for Python files and CSS files
        if not (event.src_path.endswith('.py') or event.src_path.endswith('.css')):
            return
            
        current_time = time.time()
        if current_time - self.last_reload < self.reload_delay:
            return
            
        self.last_reload = current_time
        print(f"\n🔄 File changed: {event.src_path}")
        print("🔄 Reloading Gradio app...")
        
        # Reload the module
        try:
            if self.app_module in sys.modules:
                del sys.modules[self.app_module]
            __import__(self.app_module)
            print("✅ Reload successful!")
        except Exception as e:
            print(f"❌ Reload failed: {e}")

def start_file_watcher(app_module, watch_dirs):
    """Start watching files for changes"""
    event_handler = GradioReloadHandler(app_module)
    observer = Observer()
    
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            observer.schedule(event_handler, watch_dir, recursive=True)
            print(f"👀 Watching: {watch_dir}")
    
    observer.start()
    return observer

def run_gradio_app():
    """Run the Gradio application"""
    try:
        # Import and run the demo
        from demo_v5 import demo
        print("🚀 Starting Gradio development server...")
        print("📝 Edit demo_v5.py or dark_theme.css to see changes in real-time!")
        print("🌐 Server will be available at: http://localhost:7860")
        print("🔄 Auto-reload is enabled - no need to restart manually!")
        print("\n" + "="*60)
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            enable_queue=True,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down development server...")
    except Exception as e:
        print(f"❌ Error starting Gradio app: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install gradio watchdog")

if __name__ == "__main__":
    # Directories to watch for changes
    watch_dirs = [
        str(project_root),  # Main project directory
        str(project_root / "minigpt4"),  # MiniGPT-4 source
    ]
    
    # Start file watcher in a separate thread
    observer = start_file_watcher("demo_v5", watch_dirs)
    
    try:
        # Run the Gradio app
        run_gradio_app()
    finally:
        observer.stop()
        observer.join()
