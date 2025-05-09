# Training-and-dynamics-calculations
his version has been deprecated. For the latest version, please see AdsorbML-FT!!!!!!!!!!

Fine-tuning the EquiformerV2 model to accelerate DFT calculations for CO reduction to C1 products on Cu3M1 alloys, and solving the corresponding steady-state approximation method.
# Computational Framework for CO Reduction on Cu3M1 Alloys

### 01-train: Fine-tuning EquiformerV2 Model

This folder focuses on fine-tuning the EquiformerV2 model from the Open Catalyst Project (OCP).

**Overview:**

- **Model:** EquiformerV2, a state-of-the-art deep learning model for atomistic simulations.
- **Source:** Based on the Open Catalyst Project. For the latest version of OCP, please visit the [FAIR-Chem GitHub repository](https://github.com/FAIR-Chem/fairchem). We encourage you to explore this repository for more details about the EquiformerV2 model and the Open Catalyst Project.
- **Data:** The training data is located in `01-train/01-data`. These data are molecular trajectory (traj) files calculated using Density Functional Theory (DFT).

**Fine-tuning Steps:**

To fine-tune the EquiformerV2 model, follow these steps in order:

1. **Prerequisites:**
   - Download the pre-trained EquiformerV2 checkpoint `eq2_31M_ec4_allmd.pt` and necessary libraries from the [FAIR-Chem GitHub repository](https://github.com/FAIR-Chem/fairchem). Ensure these are correctly installed and accessible in your environment.
2. **Run Training Scripts:**
   - Execute the training scripts in the following order:
     - `001`
     - `002`
     - `003`

   These scripts are designed to sequentially perform the fine-tuning process.

### 02-MK: Steady-State Approximation with SciPy

The `02-MK` folder contains Python scripts that utilize the `scipy` library to solve for steady-state approximations in chemical kinetics.

**Overview:**

- **Method:** Steady-State Approximation. This method simplifies complex kinetic models by assuming that the concentrations of intermediate species remain constant over time.
- **Library:**  [SciPy](https://scipy.org/), a Python library for scientific computing, is used for numerical solutions.
- **Application:** We have applied this approach to analyze reaction rates on 16 different Copper (Cu) alloys and pure copper.

**Data Files:**

- `All-k.csv`: This CSV file contains the rate constant (k) for all reactions considered in the kinetic model. Each row likely represents a specific reaction.
- `All-Ra`: This file contains the reaction rates (Ra) for all Carbon 1 (C1) products. It provides insights into the formation rates of different C1 products across the studied materials.

**Further Exploration:**

The scripts in `02-MK` provide a framework for analyzing reaction kinetics using steady-state approximation. You can modify and extend these scripts to investigate different reaction mechanisms, materials, and conditions.

---

