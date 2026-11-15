#!/bin/zsh
set -e
TASK_DIR="${0:A:h}"
cd "$TASK_DIR"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e '.[ai,metal]'
.venv/bin/python scripts/setup_models.py
.venv/bin/python scripts/setup_mri_model.py
.venv/bin/python scripts/setup_correction_model.py
print 'AI 与 Apple GPU 环境已安装。请重新启动 Joint2STL。'
