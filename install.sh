#!/bin/bash
################## BindProteina installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'
cuda='12.6.1'

# Define the short and long options
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for conda installation
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

### BindProteina install begin, create base environment
echo -e "Installing BindProteina environment\n"
$pkg_manager create --name BindProteina python=3.11 -y || { echo -e "Error: Failed to create BindProteina conda environment"; exit 1; }
conda env list | grep -w 'BindProteina' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'BindProteina' does not exist after creation."; exit 1; }

# Load newly created BindProteina environment
echo -e "Loading BindProteina environment\n"
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina || { echo -e "Error: Failed to activate the BindProteina environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "BindProteina" ] || { echo -e "Error: The BindProteina environment is not active."; exit 1; }
echo -e "BindProteina environment activated at ${CONDA_BASE}/envs/BindProteina"

# install required conda packages
echo -e "Instaling conda requirements\n"
# Base packages + LaProteina additions (transformers, mmseqs2)
if [ -n "$cuda" ]; then
  CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jax[cuda12-local]'==0.6.0 cuda-nvcc cudnn \
    transformers=4.48.3 mmseqs2=17.b804f \
    -c conda-forge -c nvidia -c bioconda --channel https://conda.graylab.jhu.edu -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
else
  $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jax[cuda12-local]'==0.6.0 mmseqs2=17.b804f \
    -c conda-forge -c nvidia -c bioconda --channel https://conda.graylab.jhu.edu -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
fi

conda install chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax --channel https://conda.graylab.jhu.edu -y

# make sure critical packages were installed
required_packages=(pip pandas biopython scipy tqdm jax transformers mmseqs2)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    conda list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# install pip requirements for LaProteina and upgraded BindProteina
echo -e "Installing pip requirements\n"
pip install \
    python-dotenv==1.0.1 \
    einops==0.6 \
    lightning==2.5.0 \
    loguru==0.7.2 \
    hydra-core==1.3.1 \
    biopandas==0.5.1 \
    wandb==0.17.0 \
    wget==3.2 \
    jaxtyping \
    loralib \
    'biotite>=0.41.0' \
    cpdb-protein==0.2.0 \
    'pre-commit>=4.0.0' \
    rich==14.0.0 \
    'deepdiff>=8.6.1' \
    multipledispatch \
    plotly \
    pydantic \
    rich-click \
    scikit-learn \
    xarray \
|| { echo -e "Error: Failed to install pip packages."; exit 1; }

# install specific torch and geometric versions requested for LaProteina
echo -e "Installing specialized torch and geometric versions\n"
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 || { echo -e "Error: Failed to install torch"; exit 1; }
pip install graphein==1.7.7 --no-deps || { echo -e "Error: Failed to install graphein"; exit 1; }
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html || { echo -e "Error: Failed to install torch-geometric"; exit 1; }

# install ColabDesign
echo -e "Installing ColabDesign\n"
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
conda deactivate
echo -e "BindProteina environment set up with LaProteina support\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished BindProteina installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate BindProteina\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."