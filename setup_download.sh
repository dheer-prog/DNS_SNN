#!/bin/bash
#SBATCH --job-name=dns_setup_download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00              # DNS dataset is large, give it enough time
#SBATCH --partition=cpu              # Change to your partition, cpu is fine for download
#SBATCH --qos=cpu                    # Match partition name
#SBATCH --output=logs/setup_%J.out
#SBATCH --error=logs/setup_%J.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@iitb.ac.in   # <-- change this

# =============================================================================
# CONFIGURATION — edit these paths to match your setup
# =============================================================================
PROJECT_DIR="$HOME/DNS_SNN"                        # where you cloned your repo
VENV_DIR="$HOME/dns_venv"                          # venv lives in /home — shared across all nodes
DATA_DIR="$HOME/dns_data"                          # where dataset will be downloaded
LAVA_DL_URL="https://github.com/lava-nc/lava-dl/releases/download/v0.3.2/lava_dl-0.3.2.tar.gz"

# =============================================================================
# SETUP
# =============================================================================
echo "============================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Running on    : $(hostname)"
echo "Start time    : $(date)"
echo "Project dir   : $PROJECT_DIR"
echo "Venv dir      : $VENV_DIR"
echo "Data dir      : $DATA_DIR"
echo "============================================"

# Make sure log directory exists (sbatch won't create it for you)
mkdir -p "$HOME/logs"
mkdir -p "$DATA_DIR"

# Load a sane Python — adjust version to whatever is available on Prajna
# Run `spack find python` on the login node to check available versions
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.10        # <-- adjust version as needed

# =============================================================================
# VENV — only create once; reused on every subsequent job run
# since /home is shared across ALL nodes on Prajna
# =============================================================================
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "[INFO] Virtual environment already exists at $VENV_DIR — skipping creation."
    echo "[INFO] Delete $VENV_DIR manually if you want a clean reinstall."
else
    echo ""
    echo "[STEP 1] Creating virtual environment at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    echo "[OK] venv created."

    echo ""
    echo "[STEP 2] Activating venv and upgrading pip ..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    echo ""
    echo "[STEP 3] Installing requirements.txt ..."
    pip install -r "$PROJECT_DIR/requirements.txt"

    echo ""
    echo "[STEP 4] Installing lava-dl from release tarball ..."
    pip install "$LAVA_DL_URL"

    echo ""
    echo "[STEP 5] Installing repo as editable package (ndns.pth) ..."
    # This is what the README install instructions do — adds repo root to sys.path
    python -c "
import os
from distutils.sysconfig import get_python_lib
pth = get_python_lib() + os.sep + 'ndns.pth'
with open(pth, 'a') as f:
    f.write('$PROJECT_DIR\n')
print('ndns.pth written to:', pth)
"
    echo "[OK] All packages installed."
fi

# Always activate (whether venv is new or pre-existing)
source "$VENV_DIR/bin/activate"
echo ""
echo "[INFO] Active Python : $(which python)"
echo "[INFO] Active pip    : $(which pip)"

# =============================================================================
# DATA DOWNLOAD
# =============================================================================
echo ""
echo "[STEP 6] Editing download script to point at $DATA_DIR ..."

DOWNLOAD_SCRIPT="$PROJECT_DIR/microsoft_dns/download-dns-challenge-4.sh"

# The download script has a variable for output path — we patch it on the fly
# using a temp copy so the original is untouched
TEMP_SCRIPT="/tmp/download_dns_$SLURM_JOB_ID.sh"
cp "$DOWNLOAD_SCRIPT" "$TEMP_SCRIPT"

# Replace the BLOB_NAMES or output directory line in the script.
# The original script typically has a line like: SAVE_DIR="./dns4-datasets"
# We override it to point at $DATA_DIR
sed -i "s|SAVE_DIR=.*|SAVE_DIR=\"$DATA_DIR\"|g" "$TEMP_SCRIPT"

chmod +x "$TEMP_SCRIPT"

echo "[STEP 7] Running DNS download script ..."
echo "[INFO] Data will be saved to: $DATA_DIR"
cd "$DATA_DIR"
bash "$TEMP_SCRIPT"

DOWNLOAD_EXIT=$?
if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo "[ERROR] Download script exited with code $DOWNLOAD_EXIT"
    exit $DOWNLOAD_EXIT
fi

echo ""
echo "============================================"
echo "All done!"
echo "End time : $(date)"
echo "Data is at: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract: find $DATA_DIR -name '*.tar.bz2' | xargs -I{} tar -xjf {} -C $DATA_DIR"
echo "  2. Synthesize training data per README"
echo "============================================"
