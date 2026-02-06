import subprocess
import sys
import importlib.util
from pathlib import Path


REQUIREMENTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-learn", "sklearn"),     
    ("streamlit", "streamlit"),
    ("astroquery", "astroquery"),
    ("imbalanced-learn", "imblearn"), 
    ("optuna", "optuna"),
]

def is_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def pip_install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def find_datanova_entrypoint() -> str:
    """Find a DataNova python file in the same directory as this installer."""
    here = Path(__file__).resolve().parent

    preferred = here / "datanova.py"
    if preferred.exists():
        return preferred.name

    candidates = sorted(here.glob("datanova*.py"))
    if candidates:
        return candidates[0].name

    fallback = here / "datanova (7).py"
    if fallback.exists():
        return fallback.name

    return "datanova.py"


def main():
    print("DataNova setup: checking/installing packages...\n")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception:
        pass

    for pip_name, import_name in REQUIREMENTS:
        if is_installed(import_name):
            print(f"✔ {pip_name} already installed (import: {import_name})")
            continue

        print(f"➜ Installing {pip_name} (import: {import_name}) ...")
        try:
            pip_install(pip_name)
        except subprocess.CalledProcessError as e:
            print(f"\n✖ FAILED installing {pip_name}. Error:\n{e}\n")
            print("Try running this manually:")
            print(f'  "{sys.executable}" -m pip install {pip_name}')
            print("\nPress Enter to close this window...")
            input()
            sys.exit(1)

        if not is_installed(import_name):
            print(f"\n✖ Installed {pip_name}, but Python still can't import '{import_name}'.")
            print("This usually means you're installing into a different Python environment.")
            print("Try running the installer with the same Python you use for Streamlit.")
            print("\nPress Enter to close this window...")
            input()
            sys.exit(1)

        print(f"✔ Installed {pip_name}")

    entry = find_datanova_entrypoint()

    print("\nAll packages are ready.")
    print("Run DataNova with:\n")
    print(f'  streamlit run "{entry}"')
    print("\nPress Enter to close this window...")
    input()


if __name__ == "__main__":
    main()
