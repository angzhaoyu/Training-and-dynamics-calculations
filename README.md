# Training-and-dynamics-calculations
Fine-tuning the EquiformerV2 model to accelerate DFT calculations for CO reduction to C1 products on Cu3M1 alloys, and solving the corresponding steady-state approximation method.
# Computational Framework for CO Reduction on Cu3M1 Alloys

## Directory Structure

### 01-train/ - Fine-tuning EquiformerV2 for Catalyst Modeling
Contains implementation for fine-tuning the EquiformerV2 architecture from the [Open-Catalyst Project](https://github.com/FAIR-Chem/fairchem).

#### Key Components:
- **01-data/**  
  DFT-calculated trajectory files for surface adsorption configurations
  - Contains `*.traj` files from VASP simulations
  - Coverage ranges: 0.25-1.0 ML for CO and intermediates

- **Training Steps** (Execute in sequence):
  1. `001_data_preprocessing.py`  
     - Converts traj files to PyG-compatible graphs
     - Applies rotational/translational invariance
  2. `002_model_finetuning.py`  
     - Loads pre-trained `eq2_31M_ec4_allmd.pt` weights
     - Implements cosine annealing (lr: 1e-4 → 1e-6) 
     - Batch size: 32 (4×A100 GPUs)
  3. `003_validation.py`  
     - Cross-validation with MAE/RSME metrics
     - Generates parity plots for adsorption energies

#### Prerequisites:
```bash
# Clone fairchem repository
git clone https://github.com/FAIR-Chem/fairchem.git

# Download pre-trained model (requires FAIR-Chem access)
wget https://dl.fbaipublicfiles.com/fairchem/models/eq2_31M_ec4_allmd.pt
