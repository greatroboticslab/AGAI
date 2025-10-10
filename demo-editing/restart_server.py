#!/usr/bin/env python3
"""
Simple server restart script
Run this after making changes to ui_components.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def restart_server():
    """Restart the demo_dev.py server"""
    print("🔄 Restarting server...")
    
    # Kill existing server
    try:
        subprocess.run(["pkill", "-f", "demo_dev"], check=False)
        time.sleep(2)
    except:
        pass
    
    # Start new server
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Starting server...")
    subprocess.Popen([
        sys.executable, "demo-editing/demo_dev.py"
    ])
    
    print("✅ Server restarted! Check http://localhost:7861")

if __name__ == "__main__":
    restart_server()
