#!/usr/bin/env python3
"""
Quick start script for MiniGPT-4 UI development
Run this to start the development server with hot reloading
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the development server"""
    
    print("🌿 MiniGPT-4 Plant Diagnostic System - Development Mode")
    print("=" * 60)
    print("🚀 Starting development server with hot reloading...")
    print("📝 Edit the UI files and see changes in real-time!")
    print("=" * 60)
    
    # Get the project root (go up one level from demo-editing folder)
    project_root = Path(__file__).parent.parent
    
    # Check if we're in the right directory
    if not (project_root / "demo-editing" / "demo_dev.py").exists():
        print("❌ Error: demo_dev.py not found!")
        print("💡 Make sure you're running this from the MiniGPT-4 project root")
        return
    
    # Check for required files
    required_files = [
        "demo-editing/demo_dev.py",
        "demo-editing/ui_components.py", 
        "demo-editing/custom_styles.css"
    ]
    
    missing_files = [f for f in required_files if not (project_root / f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("💡 Make sure all development files are present")
        return
    
    try:
        # Start the development server
        print("🔥 Starting hot-reload development server...")
        print("📱 The UI will be available at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the server")
        print("\n" + "=" * 60)
        
        # Run the advanced development server
        subprocess.run([sys.executable, "demo-editing/dev_server_advanced.py"], cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 Development server stopped!")
    except Exception as e:
        print(f"❌ Error starting development server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Python is installed")
        print("   2. Install dependencies: pip install gradio watchdog")
        print("   3. Check that all files are present")

if __name__ == "__main__":
    main()
