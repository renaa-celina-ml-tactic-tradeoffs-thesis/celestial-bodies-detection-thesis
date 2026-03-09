#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  setup_and_run.sh
#  One-command script to set up the environment and run the app.
#  Usage:  bash setup_and_run.sh
# ─────────────────────────────────────────────────────────────

set -e  # exit immediately if any command fails

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

echo "========================================"
echo "  Celestial Bodies Detection - Setup"
echo "========================================"

# ── 1. Decide which Python to use ────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3 first."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# ── 2. Create a virtual environment (if not already there) ───
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment at .venv ..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo ""
    echo "Virtual environment already exists, skipping creation."
fi

# ── 3. Activate the virtual environment ──────────────────────
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# ── 4. Upgrade pip silently, then install requirements ───────
echo ""
echo "Installing dependencies from requirements.txt ..."
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS" --quiet
echo "Dependencies installed."

# ── 5. Run the Flask app ──────────────────────────────────────
echo ""
echo "Starting the app — visit http://127.0.0.1:5000 in your browser."
echo "(Press Ctrl+C to stop)"
echo ""
cd "$PROJECT_DIR"
$PYTHON run.py
