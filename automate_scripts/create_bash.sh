#!/bin/bash

ENVIRONMENT_FILE="conda_env_fast_setup.yml"
ENVIRONMENT_NAME="daraconda"

# Function to create or update the environment and install packages
install_conda_packages() {
  local package_list=("$@")
  echo "Attempting to install: ${package_list[*]}"
  conda install -n "$ENVIRONMENT_NAME" -c conda-forge -c nvidia -c defaults -c pytorch "${package_list[@]}"
  if [ $? -eq 0 ]; then
    echo "Successfully installed: ${package_list[*]}"
  else
    echo "Error installing: ${package_list[*]}"
    echo "Please review the error messages above and consider addressing them before proceeding."
  fi
}

# --- Step 1: Create the base environment ---
echo "--- Step 1: Creating the base Conda environment: $ENVIRONMENT_NAME ---"
conda env create -f "$ENVIRONMENT_FILE" --name "$ENVIRONMENT_NAME"
if [ $? -eq 0 ]; then
  echo "Base environment '$ENVIRONMENT_NAME' created successfully."
else
  echo "Error creating the base environment. Please check your '$ENVIRONMENT_FILE'."
  exit 1 # Exit if the base environment creation fails
fi

# --- Step 2: Activate the environment ---
echo "--- Step 2: Activating the environment ---"
conda activate "$ENVIRONMENT_NAME"

# --- Step 3: Install essential build tools ---
echo "--- Step 3: Installing essential build tools ---"
install_conda_packages compilers make cmake

# --- Step 4: Install core SciPy stack ---
echo "--- Step 4: Installing core SciPy stack ---"
install_conda_packages numpy scipy pandas matplotlib seaborn sympy statsmodels

# --- Step 5: Install GPU acceleration libraries ---
echo "--- Step 5: Installing GPU acceleration libraries ---"
install_conda_packages numba cupy cudatoolkit

# --- Step 6: Install Core Machine Learning + Deep Learning ---
echo "--- Step 6: Installing Core Machine Learning + Deep Learning ---"
install_conda_packages scikit-learn xgboost-gpu lightgbm-gpu tensorflow-gpu=2.14.0 keras-gpu=2.14.0 pytorch=2.2.0 torchvision=0.17.0 cudnn transformers pycaret

# --- Step 7: Install Advanced AI & ML Libraries ---
echo "--- Step 7: Installing Advanced AI & ML Libraries ---"
install_conda_packages optuna mlflow tensorboard ray dask hyperopt skorch pytorch-lightning

# --- Step 8: Install NLP + Language libraries ---
echo "--- Step 8: Installing NLP + Language libraries ---"
install_conda_packages nltk spacy gensim beautifulsoup4 requests huggingface-hub

# --- Step 9: Install Visualization & Interactive Dashboards (including VMD) ---
echo "--- Step 9: Installing Visualization & Interactive Dashboards (including VMD) ---"
install_conda_packages plotly dash bokeh ipywidgets vmd

# --- Step 10: Install Jupyter Ecosystem ---
echo "--- Step 10: Installing Jupyter Ecosystem ---"
install_conda_packages notebook jupyterlab ipykernel

# --- Step 11: Install Web APIs & Dev Tools ---
echo "--- Step 11: Installing Web APIs & Dev Tools ---"
install_conda_packages fastapi flask

# --- Step 12: Install I/O + Data Manipulation libraries ---
echo "--- Step 12: Installing I/O + Data Manipulation libraries ---"
install_conda_packages openpyxl xlrd pyarrow h5py

# --- Step 13: Install Molecular Dynamics / Simulation libraries ---
echo "--- Step 13: Installing Molecular Dynamics / Simulation libraries ---"
install_conda_packages mdanalysis mdtraj openff-toolkit openbabel openmm-gpu hoomd-gpu lammps-gpu ase pymatgen parmed signac nglview gmxapi ambertools

# --- Step 14: Install pip-only packages ---
echo "--- Step 14: Installing pip-only packages ---"
pip install py3Dmol mdtraj_utils nglview-jupyterlab wandb
if [ $? -eq 0 ]; then
  echo "Successfully installed pip packages."
else
  echo "Error installing pip packages. Please review the error messages."
fi

echo "--- Installation process completed. ---"
echo "Please activate the environment using: conda activate $ENVIRONMENT_NAME"

exit 0
