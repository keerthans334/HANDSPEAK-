import os
import shutil

# Remove Python cache folders
for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d))
            print(f"Removed: {os.path.join(root, d)}")

# Remove log and temporary files
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".log") or file.endswith(".tmp"):
            os.remove(os.path.join(root, file))
            print(f"Removed: {os.path.join(root, file)}")

# Remove Jupyter Notebook checkpoints
for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == ".ipynb_checkpoints":
            shutil.rmtree(os.path.join(root, d))
            print(f"Removed: {os.path.join(root, d)}")
