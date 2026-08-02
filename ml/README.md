# Experimental machine-learning workspace

This directory contains the temporary MM-Fi and SyncWB 3D-pose experiments.
It is deliberately kept outside `src/sync_workbench` so the SyncWB package and its synchronization/export workflows do not depend on PyTorch.

## Environment

Use the existing `syncwb` Conda environment. Install the optional CUDA build of PyTorch, when required, with:

```powershell
python -m pip install -r ml/requirements-torch-cu130.txt
```

The experiment launchers are in `ml/experiments`. Each launcher changes to the repository root before running, so it can be invoked from any working directory.

## Scripts

- `mmfi_pose_quick.py`: MM-Fi training, SyncWB fine-tuning, and inference.
- `mmfi_syncwb_domain_diagnostics.py`: MM-Fi-to-SyncWB domain diagnostics.

These scripts may import SyncWB storage utilities, but nothing in
`src/sync_workbench` imports this directory.

Run `python ml/mmfi_pose_quick.py --help` or inspect the named experiment launchers for the currently recorded command lines.
