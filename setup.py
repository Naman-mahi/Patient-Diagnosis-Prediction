import subprocess
import sys
import os

def setup_environment():
    """Setup the development environment"""
    print("Setting up the healthcare prediction system...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('streamlit_env'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'streamlit_env'])
    
    # Activate virtual environment and install requirements
    if sys.platform == 'win32':
        python = '.\\streamlit_env\\Scripts\\python.exe'
        pip = '.\\streamlit_env\\Scripts\\pip.exe'
    else:
        python = './streamlit_env/bin/python'
        pip = './streamlit_env/bin/pip'
    
    print("Installing requirements...")
    subprocess.run([pip, 'install', '--upgrade', 'pip'])
    subprocess.run([pip, 'install', '-r', 'requirements.txt'])
    
    print("Setup complete!")
    print("\nTo run the application:")
    print("1. Activate the virtual environment:")
    if sys.platform == 'win32':
        print("   .\\streamlit_env\\Scripts\\activate")
    else:
        print("   source streamlit_env/bin/activate")
    print("2. Run the training script:")
    print("   python main.py")
    print("3. Start the web interface:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    setup_environment() 