# BindProteina: LaProteina Evolution & Scoring Pipeline

BindProteina is an advanced protein binder design and optimization framework. It combines the generative power of **LaProteina** with the structural accuracy of **AlphaFold2-Multimer** and the physical scoring of **PyRosetta** into a seamless, iterative evolution workflow.

## 🚀 Key Features

- **Iterative Binder Evolution**: Refine and optimize binders over multiple rounds.
- **Selective Masking**: Uses pLDDT-based confidence masking to keep stable regions fixed while regenerating low-confidence segments.
- **High-Fidelity Scoring**: Validation using AlphaFold2-Multimer v3 and PyRosetta interface analysis (dG, dSASA, Shape Complementarity).
- **Precision Targeting**: Integrated hotspot and blacklist filtering to ensure binders interact with specific target regions.
- **Interactive Visualizer**: A full-featured web-based dashboard for analyzing evolution progress, conformational changes, and metric distributions.

---

## 🧬 Workflow Overview

The pipeline operates in three distinct phases:

### 1. Initial Generation Phase
Starting from a native target PDB, the system optionally folds a linker using AF2 to provide a starting point for LaProteina. LaProteina then generates a set of candidate binders based on specified length ranges and contig constraints.

### 2. Scoring & Validation Phase
Candidate designs are scored using a robust AF2-Multimer prediction loop. The predicted complexes are then relaxed and analyzed using PyRosetta to extract interface-specific metrics like binding energy (dG) and burial (dSASA).

### 3. Iterative Evolution Phase
The best designs from the scoring phase are selected as "seeds" for the next round.
- **Monomer Masquerade**: To work with LaProteina's monomer-only architecture, binders are reformatted as single chains with the target.
- **Confidence Masking**: Regions with high pLDDT are kept fixed, while low-confidence loops or segments are masked for regeneration.
- **Regeneration**: LaProteina generates new variants for the masked regions, creating a focused trajectory of improvement.

---

## 🛠 Installation

BindProteina requires a Linux environment with a CUDA-compatible GPU (A100 recommended for high-throughput scoring).

```bash
# Clone the repository
git clone git@github.com:MaximilianMat/BindProteina.git
cd BindProteina

# Run the installation script (sets up Conda environment and AF2 weights)
bash install.sh --cuda '12.6' --pkg_manager 'conda'
```

---

## 🏃 Running the Pipeline

### Evolutionary Workflow

#### Using Settings Files (Recommended)
For long-running evolution experiments, use a settings file to avoid lengthy command-line arguments:

```bash
# Using a settings file
python laproteina/proteinfoundation/evolve.py \
    --settings settings_evolution/CLD18.json
```

Create your settings file by copying `settings_evolution/template.json` and customizing the paths and parameters:

```json
{
    "target_pdb": "/path/to/target.pdb",
    "input_csv": "/path/to/previous_round/scored/scoring_summary.csv",
    "seeds_dir": "/path/to/previous_round/scored/af2_predictions",
    "checkpoint": "/path/to/laproteina/checkpoints/LD5_motif_idx_tip.ckpt",
    "output_base_dir": "./evolution_output",
    "rounds": 24,
    "top_n": 8,
    "metric": "i_pTM",
    "target_hotspot_residues": "28-74,157-174",
    "target_blacklist_residues": "1-23,81-150"
}
```

**Note:** Command-line arguments override settings file values, allowing you to quickly adjust parameters without editing the file.

#### Using Command-Line Arguments
Alternatively, you can specify all parameters directly:

```bash
python laproteina/proteinfoundation/evolve.py \
    --target_pdb ./settings_target/PDL1.pdb \
    --input_csv ./initial_designs/scoring_summary.csv \
    --seeds_dir ./initial_designs \
    --checkpoint ./checkpoints/laproteina.ckpt \
    --rounds 5 \
    --top_n 3 \
    --metric i_pTM
```

#### Running with SLURM
Submit evolution jobs using the provided SLURM script:

```bash
sbatch laproteina_evolution.slurm \
    --settings settings_evolution/CLD18.json
```

### Pure Scoring
To score an existing directory of PDB files:

```bash
python laproteina_scoring/score.py \
    --input_dir ./my_designs \
    --output_dir ./scoring_results \
    --settings ./settings_target/PDL1.json
```

---

## 📊 LaProteina Evolution Viewer

Analyze your results using the interactive web dashboard.

1. **Start the Backend**:
   ```bash
   python visualizer/backend/server.py --evolution-dir ./evolution_output
   ```

2. **Launch the Frontend**:
   Open `visualizer/frontend/index.html` in your browser.

**Features**:
- Interactive 3D structural comparison (AF2 vs. LaProteina vs. Relaxed).
- Round-by-round metric histograms (i_pTM, pLDDT, Binder Length).
- Evolution animations to visualize conformational convergence.
- Comprehensive scoring table with filtering and sorting.
- Run Notes feature for tracking design observations.

---

## 📖 Project Documentation

- `workflow_diagram.md`: Detailed Mermaid diagrams of the pipeline logic.
- `settings_target/`: Example JSON configurations for different targets.
- `settings_evolution/`: Evolution workflow settings files (use these instead of long command-line arguments).
- `params/`: AlphaFold2 model weights storage.

---

## ⚖️ Credits & Acknowledgements

BindProteina is inspired by and builds upon several pioneer projects:
- **BindCraft**: The foundational framework for binder design pipelines by Martin Pacesa et al.
- **LaProteina**: The generative engine for rapid protein structure generation.
- **ColabDesign**: AF2-based design protocols by Sergey Ovchinnikov.
- **ProteinMPNN**: Fast sequence design by Justas Dauparas et al.
- **PyRosetta**: Structural scoring and relaxation.

---

*This project is part of a research effort to push the boundaries of de-novo protein binder design.*
