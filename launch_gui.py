#!/usr/bin/env python3
"""
QuantSim GUI Launcher
Launch the QuantSim backtesting GUI application.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui_app import main
    
    if __name__ == "__main__":
        print("🚀 Launching QuantSim GUI...")
        print("=" * 50)
        print("Professional Backtesting Engine")
        print("Features:")
        print("  ✅ Interactive strategy configuration")
        print("  ✅ Real-time backtesting")
        print("  ✅ Professional performance metrics")
        print("  ✅ Trade log analysis")
        print("  ✅ Export capabilities")
        print("=" * 50)
        
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease ensure you're in the correct directory and all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error launching GUI: {e}")
    sys.exit(1)
