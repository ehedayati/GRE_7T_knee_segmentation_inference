# Publish the standalone inference repository

Use the CONTENTS of segmentation_release as the new repository root. The new
repository contains the UI, inference/postprocessing code, launchers, manifests,
tests and documentation. Trained checkpoint files are kept separately.

Create an empty remote repository for this application. Choose a new local
checkout directory outside auto_seg_paper so the research repository and its
history are not included.

On the Linux server:

```bash
SOURCE="/home/naxos3-raid1/heday015/Jutta/Segmentation/auto_seg_paper/segmentation_release"
DEST="/home/naxos3-raid1/heday015/Jutta/Segmentation/knee_segmentation_inference"

# DEST must be a new directory; choose a different name if it already exists.
mkdir "$DEST"
rsync -av --exclude='.git/' --exclude='__pycache__/' \
  --exclude='*.py[cod]' --exclude='*.pt' --exclude='*.pth' \
  --exclude='*.ckpt' --exclude='*.pt2' --exclude='*.onnx' \
  "$SOURCE/" "$DEST/"

cd "$DEST"
git init
git add .
git diff --cached --stat

# This must print nothing: no trained weights should be staged.
git diff --cached --name-only -- '*.pt' '*.pth' '*.ckpt' '*.pt2' '*.onnx'
```

Review the staged file list, then commit and push. Replace
YOUR_NEW_REPOSITORY_URL with the URL of the empty inference repository:

```bash
git commit -m "Initial standalone knee segmentation inference release"
git branch -M main
git remote add origin YOUR_NEW_REPOSITORY_URL
git push -u origin main
```

The supplied .gitignore excludes checkpoints while retaining Python network
definitions and Bash launchers. Keep torch_unet.py, the manifests and model
checksum files in Git. Original checkpoints stay in SOURCE and are not copied
into the new checkout by the command above.

After cloning, install dependencies from README.md, obtain the checkpoint files
separately and place them at the documented relative paths. The application can
then be started from the checkout root with `python start_ui.py`.
