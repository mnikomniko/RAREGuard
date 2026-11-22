"""
Run once in Google Colab (paste into a cell) to add the files that exist only
on Drive, the fixed splits and the saved score files, to this package. It then
checks the package and downloads RAREGuard-v2.0.0-complete.zip.
"""
import os, shutil, subprocess, sys, zipfile
from google.colab import drive, files

drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/PhD/RAREGuard_v6"     # your project folder
PKG = "/content/RAREGuard-v2.0.0"

print("Choose RAREGuard-v2.0.0.zip")
name = next(iter(files.upload()))
zipfile.ZipFile(name).extractall("/content")

# fixed splits
for fn in sorted(os.listdir(os.path.join(DRIVE, "data", "splits"))):
    shutil.copy2(os.path.join(DRIVE, "data", "splits", fn), os.path.join(PKG, "data", "splits", fn))
    print("added data/splits/" + fn)

# saved scores (E1 and E9)
for exp in sorted(os.listdir(os.path.join(DRIVE, "results"))):
    src = os.path.join(DRIVE, "results", exp, "scores")
    if os.path.isdir(src):
        dst = os.path.join(PKG, "results", exp, "scores")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"added results/{exp}/scores/ ({len(os.listdir(dst))} files)")

subprocess.run([sys.executable, os.path.join(PKG, "tools", "verify_results.py")], check=True)

out = "/content/RAREGuard-v2.0.0-complete.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for dirpath, dirs, names in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in ("_archive", "__pycache__", ".ipynb_checkpoints")]
        for n in names:
            p = os.path.join(dirpath, n)
            z.write(p, os.path.relpath(p, "/content"))
print(round(os.path.getsize(out) / 2**20, 1), "MB")
files.download(out)
