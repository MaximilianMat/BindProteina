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

echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"

############################################################################################################
################## initialisation
SECONDS=0

install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

echo -e "Installing BindProteina environment\n"
$pkg_manager create --name BindProteina python=3.11 -y || { echo -e "Error: Failed to create BindProteina conda environment"; exit 1; }
conda env list | grep -w 'BindProteina' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'BindProteina' does not exist after creation."; exit 1; }

echo -e "Loading BindProteina environment\n"
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina || { echo -e "Error: Failed to activate the BindProteina environment."; exit 1; }
[ "$CONDA_DEFAULT_ENV" = "BindProteina" ] || { echo -e "Error: The BindProteina environment is not active."; exit 1; }
echo -e "BindProteina environment activated at ${CONDA_BASE}/envs/BindProteina"

echo -e "Instaling conda requirements\n"

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

################## ✅ Isolated Protenix Environment
# Protenix 2.0.0 requires NumPy 2.x, which conflicts with LaProteina.
# We install it in a dedicated environment used only for the prediction step.
echo -e "Creating specialized Protenix environment to avoid NumPy conflicts\n"
$pkg_manager create --name protenix_env python=3.11 -y || { echo -e "Error: Failed to create protenix_env"; exit 1; }
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/protenix_env
pip install protenix || { echo -e "Error: Failed to install protenix in its own env"; exit 1; }
conda deactivate

echo -e "Installing specialized torch and geometric versions for BindProteina\n"
# Reactivate main environment
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindProteina
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 || { echo -e "Error: Failed to install torch"; exit 1; }
pip install graphein==1.7.7 --no-deps || { echo -e "Error: Failed to install graphein"; exit 1; }
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html || { echo -e "Error: Failed to install torch-geometric"; exit 1; }

echo -e "Installing ColabDesign\n"
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }

echo -e "BindProteina environment set up with LaProteina and Isolated Protenix support\n"

conda deactivate

echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y

t=$SECONDS 
echo -e "Successfully finished BindProteina installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate BindProteina\""
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."