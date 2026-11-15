#!/bin/zsh
set -e
TASK_DIR="${0:A:h}"
cd "$TASK_DIR"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import PySide6.QtWidgets,pydicom,scipy,skimage,trimesh,vtk,psutil,joint2stl' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -e '.[ai,metal]'
  .venv/bin/python scripts/setup_models.py
  .venv/bin/python scripts/setup_mri_model.py
fi
exec .venv/bin/python -m joint2stl "$@"
