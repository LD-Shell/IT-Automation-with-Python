#!/usr/bin/env bash
# check_cuda.sh   —   GPU-Python + RAPIDS sanity check & optional conda install (interactive)

# --- Configuration ---
PYTHON_EXEC="python3"
# Default channel for non-RAPIDS specific installs (like CuPy/Numba if RAPIDS isn't needed)
# NOTE: For RAPIDS, specific channels ('rapidsai', 'conda-forge', 'nvidia') will be used below.
CONDA_CHANNEL="conda-forge"

# --- Helpers ---
check_python_exec() {
    if ! command -v $PYTHON_EXEC &> /dev/null; then
        echo "⚠️ '$PYTHON_EXEC' not found, trying 'python'..."
        PYTHON_EXEC="python"
        if ! command -v $PYTHON_EXEC &> /dev/null; then
            echo "❌ No Python executable found in PATH."
            echo "   Ensure you are in your desired Conda environment and Python is installed."
            return 1
        fi
    fi
    echo "ℹ️ Using Python: $($PYTHON_EXEC --version) (Path: $(command -v $PYTHON_EXEC))"
    return 0
}

check_conda_exec() {
    if ! command -v conda &> /dev/null; then
        echo "❌ 'conda' command not found. Cannot manage packages."
        echo "   Ensure Conda is installed and initialized in your shell (e.g., 'conda init bash')."
        return 1
    fi
    echo "ℹ️ Using conda: $(command -v conda)"
    echo "ℹ️ Active env: ${CONDA_DEFAULT_ENV:-none (likely base)}"
    if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "⚠️ Warning: No Conda environment seems to be active. Installs will target the 'base' env unless specified."
    fi
    return 0
}

prompt_install() {
    local pkgs_to_install=("$@")
    echo -e "\n🛠️ The following packages (plus dependencies like cudatoolkit) will be installed/updated:"
    for p in "${pkgs_to_install[@]}"; do
        echo "   • $p"
    done

    # Determine channels based on whether RAPIDS is included
    local install_cmd_preview
    local install_rapids_flag=$1 # Check if the *first* argument indicates RAPIDS install intent (see run_install logic)
    if (( install_rapids_flag )); then
         install_cmd_preview="conda install -y -c rapidsai -c conda-forge -c nvidia ..."
         echo "   Using channels: rapidsai, conda-forge, nvidia (Recommended for RAPIDS)"
    else
         install_cmd_preview="conda install -y -c $CONDA_CHANNEL -c nvidia ..."
         echo "   Using channels: $CONDA_CHANNEL, nvidia"
    fi

    echo -n "Proceed with installation? [y/N] "
    read -r ans
    [[ "$ans" =~ ^[Yy] ]] && return 0
    echo "ℹ️ Installation skipped by user."
    return 1
}

run_install() {
    # Pass flags: install_cupy, install_numba, install_rapids
    local install_cupy=$1 install_numba=$2 install_rapids=$3
    local pkgs=()
    local conda_channels=()
    local rapids_install_intent=0 # Flag to pass to prompt_install

    (( install_cupy ))  && pkgs+=("cupy")
    (( install_numba )) && pkgs+=("numba")
    if (( install_rapids )); then
        pkgs+=("cudf" "cuml" "cugraph")
        # Use RAPIDS recommended channels when RAPIDS components are involved
        conda_channels=("-c" "rapidsai" "-c" "conda-forge" "-c" "nvidia")
        rapids_install_intent=1
    else
        # Use the default channel + nvidia for CuPy/Numba if RAPIDS isn't being installed
        conda_channels=("-c" "$CONDA_CHANNEL" "-c" "nvidia")
    fi

    if (( ${#pkgs[@]} == 0 )); then
        echo "ℹ️ Nothing specific marked for installation."
        return 0
    fi

    # Always add cudatoolkit for GPU libraries managed by conda
    pkgs+=("cudatoolkit")

    # Pass the rapids_install_intent flag first to prompt_install for preview message
    if ! prompt_install $rapids_install_intent "${pkgs[@]}"; then
        return 1
    fi

    echo "🔧 Running: conda install -y ${conda_channels[*]} ${pkgs[*]}"
    conda install -y "${conda_channels[@]}" "${pkgs[@]}" \
        && echo "✅ Install command finished successfully." \
        || { echo "❌ Install command failed."; return 1; }

    return 0
}

# --- Main ---
echo "🚀 Starting GPU/Python/RAPIDS Check..."

check_python_exec || exit 1

echo -e "\n🔍 Checking NVIDIA Driver & GPU"
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 'nvidia-smi' not found."
    echo "   Please ensure NVIDIA drivers are installed correctly and nvidia-smi is in your PATH."
    exit 1
fi
echo "   Driver/GPU Info:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || {
    echo "❌ Failed to query GPU via nvidia-smi."
    exit 1
} | sed 's/^/    /' # Indent output

# --- Initialize Status ---
install_cupy=0
install_numba=0
install_rapids=0
missing_summary=() # Store user-friendly names of missing components

# --- CuPy check ---
echo -e "\n🔍 CuPy Status"
# Use temporary variable to avoid masking $? later
cupy_output_and_exit_code=$(
    $PYTHON_EXEC - <<'PYCODE'
import sys
try:
    import cupy as cp
    ver = cp.__version__
    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count == 0:
        print(f"⚠️ CuPy {ver} found, but no CUDA devices detected by CuPy.")
        sys.exit(1) # Treat as failure for install prompt
    else:
        print(f"✅ CuPy {ver} found and {dev_count} CUDA device(s) detected.")
        sys.exit(0)
except ImportError:
    print("❌ CuPy not found or import error.")
    sys.exit(1)
except Exception as e:
    print(f"❌ CuPy error: {e}")
    sys.exit(1)
PYCODE
    echo $? # Append exit code after output
)
cupy_exit_code="${cupy_output_and_exit_code##*$'\n'}" # Extract last line (exit code)
cupy_info="${cupy_output_and_exit_code%$'\n'*}"      # Extract output before last line
echo "$cupy_info"
if [[ "$cupy_exit_code" -ne 0 ]]; then
    install_cupy=1
    missing_summary+=("CuPy")
fi

# --- Numba-CUDA check ---
echo -e "\n🔍 Numba (CUDA) Status"
numba_output_and_exit_code=$(
    $PYTHON_EXEC - <<'PYCODE'
import sys
try:
    from numba import cuda
    if cuda.is_available():
        # Try getting device count for a more thorough check
        try:
            devices = cuda.list_devices()
            if not devices:
                 print("⚠️ Numba CUDA available, but no devices found by Numba.")
                 sys.exit(1) # Treat as failure for install prompt
            else:
                print(f"✅ Numba CUDA available ({len(devices)} device(s)).")
                sys.exit(0)
        except Exception as e:
             print(f"⚠️ Numba CUDA seems available, but device check failed: {e}")
             sys.exit(1) # Treat as failure
    else:
        print("❌ Numba found, but CUDA support is not available/enabled.")
        sys.exit(1)
except ImportError:
    print("❌ Numba not found or import error.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Numba CUDA check error: {e}")
    sys.exit(1)
PYCODE
    echo $?
)
numba_exit_code="${numba_output_and_exit_code##*$'\n'}"
numba_info="${numba_output_and_exit_code%$'\n'*}"
echo "$numba_info"
if [[ "$numba_exit_code" -ne 0 ]]; then
    install_numba=1
    missing_summary+=("Numba (with CUDA)")
fi

# --- RAPIDS check ---
echo -e "\n🔍 RAPIDS (cuDF, cuML, cuGraph) Status"
rapids_output_and_exit_code=$(
    $PYTHON_EXEC - <<'PYCODE'
import sys
pkgs_found = []
errs = []
pkg_versions = {}

try:
    import cudf
    pkgs_found.append("cuDF")
    pkg_versions["cuDF"] = cudf.__version__
except Exception as e:
    errs.append(f"cuDF: {e}")

try:
    import cuml
    pkgs_found.append("cuML")
    pkg_versions["cuML"] = cuml.__version__
except Exception as e:
    errs.append(f"cuML: {e}")

try:
    import cugraph
    pkgs_found.append("cuGraph")
    pkg_versions["cuGraph"] = cugraph.__version__
except Exception as e:
    errs.append(f"cuGraph: {e}")

if not pkgs_found:
    print("❌ RAPIDS core libraries (cuDF, cuML, cuGraph) not found.")
    for e in errs: print(f"   • Error detail: {e}")
    sys.exit(1)
elif errs:
    print(f"⚠️ RAPIDS partial install / issues detected.")
    for p in pkgs_found: print(f"   ✅ {p} {pkg_versions.get(p, '')} found.")
    for e in errs: print(f"   ❌ Error importing/finding: {e}")
    sys.exit(1) # Treat partial/errors as needing potential fix
else:
    print(f"✅ RAPIDS core libraries found:")
    for p in pkgs_found: print(f"   • {p} {pkg_versions.get(p, '')}")
    sys.exit(0)
PYCODE
    echo $?
)
rapids_exit_code="${rapids_output_and_exit_code##*$'\n'}"
rapids_info="${rapids_output_and_exit_code%$'\n'*}"
echo "$rapids_info"
if [[ "$rapids_exit_code" -ne 0 ]]; then
    install_rapids=1
    missing_summary+=("RAPIDS (cuDF/cuML/cuGraph)")
fi

# --- Interactive install phase ---
if (( install_cupy || install_numba || install_rapids )); then
    echo -e "\n❗️ Found issues with: ${missing_summary[*]}"
    check_conda_exec || exit 1 # Check conda only if needed for install
    # Pass the boolean flags to run_install
    run_install $install_cupy $install_numba $install_rapids || {
      echo -e "\n❌ Installation process failed or was skipped."
      echo "   You may need to install/fix the packages manually."
      exit 1
    }
    echo -e "\n✨ Attempted install/fix. Please re-run this script to verify the environment."
else
    echo -e "\n🎉 All checks passed! CuPy, Numba-CUDA, and RAPIDS seem correctly installed and configured."
fi

echo -e "\n🏁 Check finished."
exit 0