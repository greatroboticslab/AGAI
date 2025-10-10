#!/usr/bin/env python3
"""
Test script to verify hot reload functionality
This script will make changes to ui_components.py to test live reloading
"""

import time
import os
from pathlib import Path

def test_hot_reload():
    """Test the hot reload functionality by modifying ui_components.py"""
    
    print("🧪 Testing Hot Reload Functionality")
    print("=" * 50)
    
    # Path to ui_components.py
    ui_file = Path(__file__).parent / "ui_components.py"
    
    if not ui_file.exists():
        print("❌ ui_components.py not found!")
        return False
    
    # Read the current content
    with open(ui_file, 'r') as f:
        content = f.read()
    
    # Find the test print statement
    test_line = "🔥 HOT RELOAD: create_header() function loaded at"
    
    if test_line not in content:
        print("❌ Test line not found in ui_components.py")
        return False
    
    print("✅ Test line found in ui_components.py")
    print("🔄 Making test changes...")
    
    # Make a small change to trigger reload
    new_test_line = f"🔥 HOT RELOAD: create_header() function loaded at {time.strftime('%H:%M:%S')} - TEST MODIFIED"
    
    # Replace the test line
    modified_content = content.replace(test_line, new_test_line)
    
    # Write the modified content
    with open(ui_file, 'w') as f:
        f.write(modified_content)
    
    print("✅ Modified ui_components.py")
    print("🔄 Waiting 3 seconds for hot reload...")
    time.sleep(3)
    
    # Restore the original content
    with open(ui_file, 'w') as f:
        f.write(content)
    
    print("✅ Restored original ui_components.py")
    print("🎉 Hot reload test completed!")
    print("\n💡 If you see '🔄 Restarting server...' messages above,")
    print("   then hot reload is working correctly!")
    
    return True

if __name__ == "__main__":
    test_hot_reload()
