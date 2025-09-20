#!/usr/bin/env python3
"""
QuantSim Web App Launcher
Launch the QuantSim Streamlit web application.
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_streamlit_installed():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Launching QuantSim Web Application...")
    print("=" * 60)
    print("🌐 Opening in your default web browser...")
    print("📊 Professional backtesting interface loading...")
    print("🎯 Interactive charts and real-time analysis ready!")
    print("=" * 60)
    
    # Launch Streamlit
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 QuantSim Web App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    print("🚀 QuantSim Web Application Launcher")
    print("=" * 50)
    
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        print("📦 Streamlit not found. Installing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please run:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Launch the app
    launch_streamlit()

if __name__ == "__main__":
    main()
