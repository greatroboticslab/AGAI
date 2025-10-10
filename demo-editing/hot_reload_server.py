#!/usr/bin/env python3
"""
True Hot Reload Server for MiniGPT-4 Gradio UI
Features:
- Process restarting (not just re-importing)
- Clear logging with emojis
- Automatic browser refresh
- Clean process management
"""

import os
import sys
import time
import signal
import subprocess
import threading
import psutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class HotReloadHandler(FileSystemEventHandler):
    """Handles file changes and restarts the Gradio process"""
    
    def __init__(self, server_manager):
        self.server_manager = server_manager
        self.last_reload = 0
        self.reload_delay = 2.0  # Minimum delay between reloads (increased for stability)
        self.pending_restart = False
        
    def _handle_file_change(self, event):
        """Handle file change events"""
        if event.is_directory:
            return
            
        # Only watch Python and CSS files
        if not (event.src_path.endswith('.py') or event.src_path.endswith('.css')):
            return
            
        # Skip __pycache__ and other non-essential files
        if '__pycache__' in event.src_path or event.src_path.endswith('.pyc'):
            return
            
        current_time = time.time()
        if current_time - self.last_reload < self.reload_delay:
            return
            
        # If there's already a pending restart, just update the file info
        if self.pending_restart:
            file_name = Path(event.src_path).name
            print(f"📝 Additional change detected: {file_name}")
            return
            
        self.last_reload = current_time
        self.pending_restart = True
        
        # Determine file type for logging
        file_type = "🎨 CSS" if event.src_path.endswith('.css') else "🐍 Python"
        file_name = Path(event.src_path).name
        
        print(f"\n{file_type} file changed: {file_name}")
        print("🔄 Restarting server...")
        
        # Restart the server
        self.server_manager.restart_server()
        
        # Reset pending restart after a delay
        def reset_pending():
            time.sleep(1)
            self.pending_restart = False
        threading.Thread(target=reset_pending, daemon=True).start()
    
    def on_modified(self, event):
        self._handle_file_change(event)
    
    def on_created(self, event):
        self._handle_file_change(event)
    
    def on_moved(self, event):
        self._handle_file_change(event)

class GradioServerManager:
    """Manages the Gradio server process with hot reloading"""
    
    def __init__(self):
        self.process = None
        self.port = 7861
        self.max_retries = 3
        self.retry_count = 0
        
    def start_server(self):
        """Start the Gradio server process"""
        try:
            # Kill any existing process on the port
            self.kill_process_on_port(self.port)
            
            print("🚀 Starting Gradio server...")
            print(f"📱 Server will be available at: http://localhost:{self.port}")
            
            # Start the demo process
            demo_path = project_root / "demo-editing" / "demo_dev.py"
            self.process = subprocess.Popen(
                [sys.executable, str(demo_path)],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start a thread to monitor the process output
            threading.Thread(target=self.monitor_output, daemon=True).start()
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            if self.process.poll() is None:
                print("✅ Server started successfully!")
                self.retry_count = 0
                return True
            else:
                print("❌ Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def restart_server(self):
        """Restart the Gradio server process"""
        print("🔄 Restarting server...")
        
        # Kill the current process
        self.kill_current_process()
        
        # Wait longer to ensure process is fully stopped
        time.sleep(2)
        
        # Start a new process
        if self.start_server():
            print("✅ Server restarted successfully!")
        else:
            print("❌ Server restart failed!")
            self.handle_restart_failure()
    
    def kill_current_process(self):
        """Kill the current Gradio process"""
        if self.process and self.process.poll() is None:
            print("🛑 Stopping current server...")
            try:
                # Try graceful termination first
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                print("⚠️  Force killing server process...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"⚠️  Error stopping process: {e}")
    
    def kill_process_on_port(self, port):
        """Kill any process running on the specified port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.info['connections'] or []:
                        if conn.laddr.port == port:
                            print(f"🛑 Killing existing process on port {port} (PID: {proc.info['pid']})")
                            proc.kill()
                            time.sleep(1)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            print(f"⚠️  Error checking port {port}: {e}")
    
    def monitor_output(self):
        """Monitor the server process output"""
        if not self.process:
            return
            
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    print(f"[SERVER] {line.rstrip()}")
        except Exception as e:
            print(f"⚠️  Error monitoring output: {e}")
    
    def handle_restart_failure(self):
        """Handle server restart failures"""
        self.retry_count += 1
        if self.retry_count < self.max_retries:
            print(f"🔄 Retry {self.retry_count}/{self.max_retries} in 3 seconds...")
            time.sleep(3)
            self.restart_server()
        else:
            print("❌ Max retries reached. Please check for errors and restart manually.")
            self.retry_count = 0
    
    def stop_server(self):
        """Stop the Gradio server"""
        print("👋 Stopping server...")
        self.kill_current_process()
        print("✅ Server stopped")

def start_file_watcher(server_manager, watch_dirs):
    """Start watching files for changes"""
    event_handler = HotReloadHandler(server_manager)
    observer = Observer()
    
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            observer.schedule(event_handler, watch_dir, recursive=True)
            print(f"👀 Watching: {watch_dir}")
    
    observer.start()
    return observer

def main():
    """Main function to run the hot reload server"""
    print("🔥 MiniGPT-4 Hot Reload Development Server")
    print("=" * 60)
    print("🚀 Starting with true process restarting...")
    print("📝 Edit any .py or .css file to see instant changes!")
    print("🔄 Server will restart automatically on file changes")
    print("=" * 60)
    
    # Create server manager
    server_manager = GradioServerManager()
    
    # Directories to watch
    watch_dirs = [
        str(project_root / "demo-editing"),
        str(project_root / "minigpt4"),
    ]
    
    # Start file watcher
    observer = start_file_watcher(server_manager, watch_dirs)
    
    try:
        # Start the server
        if server_manager.start_server():
            print("\n🎉 Development server is running!")
            print("📱 Open your browser to: http://localhost:7861")
            print("🛑 Press Ctrl+C to stop the server")
            print("\n" + "=" * 60)
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start server")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        observer.stop()
        observer.join()
        server_manager.stop_server()
    
    return 0

if __name__ == "__main__":
    # Check dependencies
    try:
        import psutil
        import watchdog
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install psutil watchdog")
        sys.exit(1)
    
    sys.exit(main())
