import subprocess
import sys
import importlib

packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "streamlit",
    "astroquery"
]

for pkg in packages:
    if importlib.util.find_spec(pkg) is None:
        print(f"{pkg} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f"{pkg} installed successfully.\n")
    else:
        print(f"{pkg} is already installed.\n")

print("All packages are ready! You can now run datanova.py with:")
print("python datanova.py")
print("\nPress Enter to close this window...")
input()
