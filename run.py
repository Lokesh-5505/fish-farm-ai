"""
MAIN DEPLOYMENT SCRIPT
Run this file to launch the Fish Farm Disease Prediction Streamlit Application

Usage:
    python run.py
    
Or via command line:
    streamlit run app/streamlit_app.py
"""

import subprocess
import os
import sys

def main():
    """Launch the Streamlit application"""
    
    print("\n" + "=" * 80)
    print("🐟 FISH FARM DISEASE PREDICTION SYSTEM - DEPLOYMENT")
    print("=" * 80)
    
    # Get the app path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, 'app', 'streamlit_app.py')
    
    print(f"\n📁 Project Root: {script_dir}")
    print(f"📁 App Path: {app_path}")
    
    # Check if app file exists
    if not os.path.exists(app_path):
        print(f"\n❌ ERROR: Application file not found at {app_path}")
        sys.exit(1)
    
    print(f"\n✅ Application file found!")
    print("\n🚀 Starting Streamlit Application...")
    print("-" * 80)
    print("\n📊 The application will open at: http://localhost:8501")
    print("🔗 Press Ctrl+C to stop the server\n")
    print("-" * 80 + "\n")
    
    try:
        # Run streamlit app
        subprocess.run(
            [sys.executable, '-m', 'streamlit', 'run', app_path],
            cwd=script_dir
        )
    except KeyboardInterrupt:
        print("\n\n" + "-" * 80)
        print("⏹️  Application stopped by user")
        print("-" * 80 + "\n")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
