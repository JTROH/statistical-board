#!/usr/bin/env bash
# Build dist/Statistical Workbench.app from a clean virtualenv.
#   packaging/build_mac_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
# pandas 3 changes dtype inference that stat_board/prep.py relies on; stay on 2.x until that is fixed.
.venv/bin/pip install -q -r requirements.txt "pandas<3" pywebview pyinstaller

.venv/bin/pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/workbench.spec

echo
echo "Built: dist/Statistical Workbench.app"
echo "Install: drag it into /Applications (or: cp -R \"dist/Statistical Workbench.app\" /Applications/)"
