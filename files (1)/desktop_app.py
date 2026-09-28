"""
desktop_app.py - Native Windows Desktop Application Launcher for DemandIQ™
Launches DemandIQ as a standalone desktop application window with full desktop features,
no browser address bar, and native OS window integration.
"""
import os
import sys
import time
import threading
import urllib.request
import subprocess

PORT = 5050
URL = f"http://127.0.0.1:{PORT}"

def is_server_running():
    try:
        with urllib.request.urlopen(URL, timeout=1.5) as res:
            return res.status == 200
    except Exception:
        return False

def run_flask_backend():
    """Runs the Flask backend in a separate background thread if not already running."""
    if is_server_running():
        print(f"[Desktop App] Backend is already running on {URL}.")
        return

    print("[Desktop App] Starting DemandIQ Flask backend server...")
    try:
        import app
        app.app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[Desktop App] Error starting backend: {e}")

def open_native_window():
    """Tries pywebview first for true native window, or falls back to Chrome/Edge app window mode."""
    # Ensure backend is up
    for _ in range(30):
        if is_server_running():
            break
        time.sleep(0.3)

    # Strategy 1: pywebview (True Native Windows Desktop App)
    try:
        import webview
        print("[Desktop App] Opening native window via pywebview...")
        webview.create_window(
            title="DemandIQ™ Enterprise — Inventory Forecasting & Optimization",
            url=URL,
            width=1440,
            height=900,
            min_size=(1024, 680),
            text_select=True,
            confirm_close=False
        )
        webview.start()
        return
    except Exception as e:
        print(f"[Desktop App] pywebview window could not start ({e}). Trying Chrome/Edge App Mode...")

    # Strategy 2: Google Chrome or Microsoft Edge in Dedicated App Mode
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    for exe in chrome_paths:
        if os.path.exists(exe):
            print(f"[Desktop App] Launching standalone app window using: {exe}")
            subprocess.Popen([exe, f"--app={URL}", "--window-size=1440,900"])
            return

    # Strategy 3: Default OS browser
    import webbrowser
    webbrowser.open(URL)

if __name__ == '__main__':
    backend_thread = threading.Thread(target=run_flask_backend, daemon=True)
    backend_thread.start()
    open_native_window()
