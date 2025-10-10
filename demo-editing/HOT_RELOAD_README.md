# 🔥 Hot Reload Development Setup

This setup provides **true hot reloading** with process restarting, clear logging, and automatic browser refresh for the MiniGPT-4 Gradio UI.

## 🚀 Quick Start

### Start Hot Reload Server
```bash
# From MiniGPT-4 project root
python demo-editing/quick_start.py
```

### Alternative: Direct Server Start
```bash
python demo-editing/hot_reload_server.py
```

## ✨ Features

### 🔄 True Process Restarting
- **Not just re-importing** - Actually restarts the Gradio process
- **Clean process management** - Kills previous processes properly
- **Port conflict resolution** - Automatically handles port 7860 conflicts

### 📝 Clear Logging
- **🔄 Restarting server...** - Clear restart messages
- **✅ Server restarted successfully!** - Success confirmations
- **❌ Server restart failed!** - Error handling
- **🛑 Stopping current server...** - Process cleanup logs

### 🌐 Automatic Browser Refresh
- **Gradio's built-in reloading** enabled
- **inbrowser=True** - Opens browser automatically
- **enable_monitoring=True** - Monitors for changes
- **show_tips=True** - Shows helpful tips

### 🎯 File Watching
- **Python files** (`.py`) - Triggers server restart
- **CSS files** (`.css`) - Triggers server restart
- **Recursive watching** - Monitors subdirectories
- **Smart filtering** - Ignores `__pycache__` and `.pyc` files

## 📁 Files Structure

```
demo-editing/
├── quick_start.py           # 🚀 One-command startup
├── hot_reload_server.py     # 🔥 Main hot reload server
├── demo_dev.py              # 🎨 Development demo
├── ui_components.py         # 🧩 UI components
├── custom_styles.css        # 🎨 CSS styling
├── test_hot_reload.py       # 🧪 Test script
└── HOT_RELOAD_README.md     # 📚 This file
```

## 🧪 Testing Hot Reload

### Test Script
```bash
python demo-editing/test_hot_reload.py
```

### Manual Testing
1. **Start the server**: `python demo-editing/quick_start.py`
2. **Edit a file**: Modify `ui_components.py` or `custom_styles.css`
3. **Watch the logs**: You should see "🔄 Restarting server..."
4. **Check browser**: The page should refresh automatically

### Test Print Statements
The following test statements are embedded in the code:
- `🔥 HOT RELOAD: create_header() function loaded at [TIME]`
- `🎨 HOT RELOAD: CSS loaded at [TIME]`

These will appear in the console when the server restarts.

## 🔧 How It Works

### 1. Process Management
```python
# Kill existing process
self.kill_current_process()

# Start new process
self.process = subprocess.Popen([...])

# Monitor output
threading.Thread(target=self.monitor_output, daemon=True).start()
```

### 2. File Watching
```python
# Watch for changes
observer.schedule(event_handler, watch_dir, recursive=True)

# On file change
def on_modified(self, event):
    print("🔄 Restarting server...")
    self.server_manager.restart_server()
```

### 3. Port Management
```python
# Kill processes on port 7860
def kill_process_on_port(self, port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        if conn.laddr.port == port:
            proc.kill()
```

## 🎯 Development Workflow

### 1. Start Development
```bash
python demo-editing/quick_start.py
```

### 2. Edit Files
- **UI Components**: Edit `ui_components.py`
- **Styling**: Edit `custom_styles.css`
- **Main Logic**: Edit `demo_dev.py`

### 3. See Changes
- **Automatic restart** on file save
- **Clear logging** in terminal
- **Browser refresh** automatically
- **No manual restart needed!**

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check dependencies
pip install psutil watchdog gradio

# Check port availability
lsof -i :7860

# Kill existing processes
pkill -f "demo_dev.py"
```

### Changes Not Appearing
1. **Check file watcher** - Look for "👀 Watching:" messages
2. **Check restart logs** - Look for "🔄 Restarting server..."
3. **Check browser** - Refresh manually if needed
4. **Check file paths** - Ensure files are in watched directories

### Process Not Restarting
1. **Check permissions** - Ensure write access to files
2. **Check file extensions** - Only `.py` and `.css` files trigger reload
3. **Check console output** - Look for error messages
4. **Check port conflicts** - Ensure port 7860 is available

## 📊 Performance

### Memory Usage
- **Low overhead** - Only watches specific directories
- **Efficient restarting** - Kills old process before starting new one
- **Smart filtering** - Ignores unnecessary files

### Response Time
- **~1 second** - File change detection delay
- **~3 seconds** - Server restart time
- **~1 second** - Browser refresh time

### Resource Management
- **Automatic cleanup** - Kills old processes
- **Port management** - Handles conflicts automatically
- **Error recovery** - Retries on failure

## 🎉 Benefits

✅ **True hot reloading** - Process restarting, not just re-importing  
✅ **Clear feedback** - Know exactly what's happening  
✅ **Automatic browser refresh** - No manual refresh needed  
✅ **Robust error handling** - Handles failures gracefully  
✅ **Port conflict resolution** - No more "port in use" errors  
✅ **Clean process management** - No zombie processes  

## 🚀 Next Steps

1. **Start developing**: `python demo-editing/quick_start.py`
2. **Edit files**: Make changes to UI components or CSS
3. **See changes**: Watch the magic happen automatically!
4. **Enjoy**: No more launch-close cycles! 🎉

Happy coding with true hot reloading! 🔥
