#!/usr/bin/env python3
"""
Quick Start Script for MiniGPT-4 UI Development with True Hot Reloading
This script starts the hot reload server with process restarting
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the hot reload development server"""
    
    print("🌿 MiniGPT-4 Plant Diagnostic System - Hot Reload Development")
    print("=" * 70)
    print("🔥 Starting TRUE hot reload server with process restarting...")
    print("📝 Edit any .py or .css file to see instant changes!")
    print("🔄 Server will restart automatically on file changes")
    print("📱 Browser will refresh automatically")
    print("=" * 70)
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Check if we're in the right directory
    if not (project_root / "demo-editing" / "hot_reload_server.py").exists():
        print("❌ Error: hot_reload_server.py not found!")
        print("💡 Make sure you're running this from the MiniGPT-4 project root")
        return 1
    
    # Check for required files
    required_files = [
        "demo-editing/hot_reload_server.py",
        "demo-editing/demo_dev.py",
        "demo-editing/ui_components.py", 
        "demo-editing/custom_styles.css"
    ]
    
    missing_files = [f for f in required_files if not (project_root / f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("💡 Make sure all development files are present")
        return 1
    
    # Check dependencies
    try:
        import psutil
        import watchdog
        import gradio
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install psutil watchdog gradio")
        return 1
    
    try:
        print("🚀 Starting hot reload development server...")
        print("📱 The UI will be available at: http://localhost:7861")
        print("🛑 Press Ctrl+C to stop the server")
        print("\n" + "=" * 70)
        
        # Start the hot reload server
        result = subprocess.run([
            sys.executable, 
            str(project_root / "demo-editing" / "hot_reload_server.py")
        ], cwd=str(project_root))
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n👋 Development server stopped!")
        return 0
    except Exception as e:
        print(f"❌ Error starting development server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Python is installed")
        print("   2. Install dependencies: pip install psutil watchdog gradio")
        print("   3. Check that all files are present")
        print("   4. Make sure port 7861 is not in use")
        return 1

if __name__ == "__main__":
    sys.exit(main())
