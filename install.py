import subprocess
import sys


def install_if_missing(package: str):
    """Install a package via pip if not already importable."""
    mod_name = package.split("==")[0].strip()
    try:
        __import__(mod_name)
        print(f"OK: {package} already installed")
    except Exception:
        print(f"Installing {package} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: pip failed for {package}: {e}")


# Read requirement.txt and try each non-empty, non-comment line
try:
    with open("requirement.txt", "r", encoding="utf-8") as f:
        for line in f:
            pkg = line.strip()
            if pkg and not pkg.startswith("#"):
                install_if_missing(pkg)
except FileNotFoundError:
    print("requirement.txt not found.")
