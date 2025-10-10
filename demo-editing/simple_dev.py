#!/usr/bin/env python3
"""
Simple Development Server for MiniGPT-4 Gradio UI
- Basic file watching
- Process restarting
- No complex dependencies
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class SimpleFileHandler(FileSystemEventHandler):
    """Simple file change handler"""
    
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_change = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only watch Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Skip __pycache__ files
        if '__pycache__' in event.src_path or event.src_path.endswith('.pyc'):
            return
            
        current_time = time.time()
        if current_time - self.last_change < 1.0:  # Debounce
            return
            
        self.last_change = current_time
        file_name = Path(event.src_path).name
        print(f"\n🐍 File changed: {file_name}")
        print("🔄 Restarting server...")
        self.restart_callback()

class SimpleDevServer:
    """Simple development server with hot reload"""
    
    def __init__(self):
        self.process = None
        self.port = 7862
        self.running = False
        
    def start_server(self):
        """Start the Gradio server"""
        try:
            print("🚀 Starting Gradio server...")
            self.process = subprocess.Popen([
                sys.executable, "demo-editing/demo_dev.py"
            ], cwd=str(project_root))
            
            # Wait for server to start
            time.sleep(5)
            
            if self.process.poll() is None:
                print("✅ Server started successfully!")
                print(f"📱 Open: http://localhost:{self.port}")
                return True
            else:
                print("❌ Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def restart_server(self):
        """Restart the server"""
        if self.process:
            print("🛑 Stopping current server...")
            self.process.terminate()
            self.process.wait()
            time.sleep(2)
        
        return self.start_server()
    
    def stop_server(self):
        """Stop the server"""
        if self.process:
            print("👋 Stopping server...")
            self.process.terminate()
            self.process.wait()
            print("✅ Server stopped")

def main():
    """Main function"""
    print("🔥 Simple MiniGPT-4 Development Server")
    print("=" * 50)
    print("📝 Edit .py files to see changes!")
    print("=" * 50)
    
    server = SimpleDevServer()
    
    # Start file watcher
    event_handler = SimpleFileHandler(server.restart_server)
    observer = Observer()
    observer.schedule(event_handler, "demo-editing", recursive=True)
    observer.start()
    
    try:
        # Start the server
        if server.start_server():
            print("\n🎉 Development server is running!")
            print("🛑 Press Ctrl+C to stop")
            print("=" * 50)
            
            # Keep running
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start server")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        observer.stop()
        server.stop_server()
        print("✅ Done!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        observer.stop()
        server.stop_server()
        return 1
    finally:
        observer.join()

if __name__ == "__main__":
    sys.exit(main())
