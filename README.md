# CNPEM/Ilum INCT - Max Planck Meeting Tutorials 2025

## 🗂️ Table of Contents

- [Overview](#overview)
- [Tutorial 1: Machine Learning Basics and ML for Electronic Structure](#tutorial-1-machine-learning-basics-and-ml-for-electronic-structure)
   - [Run the Tutorial](#run-the-tutorial)
   - [Basics-GPR-and-NN Tutorials](#basics-gpr-and-nn-tutorials)
   - [SALTED Tutorial](#salted-tutorial)
- [Tutorial 2: Nuclear Quantum Effects with i-PI](#tutorial-2-nuclear-quantum-effects-with-i-pi)
   - [Overview](#overview-1)
   - [Resources](#resources)
   - [Installation](#installation)
- [Tutorial 3: Machine Learning Interatomic Potentials – Active Learning](#tutorial-3-machine-learning-interatomic-potentials--active-learning)
   - [Overview](#overview-2)
   - [Resources](#resources-1)
   - [Installation](#installation-1)
   - [References](#references-1)
- [Useful Links](#useful-links)


---

## Overview

Collection of tutorials from the **SabIA group** presented at the **2025 INCT - Max Planck Meeting on Electronic Structure Methods and Materials Informatics** at **CNPEM (Campinas, Brazil)**. A YouTube link to the timestamped part of each tutorial, including the introductory lectures, is provided from the official livestream recording of the event. Slides from the lectures can be found in the respective folders.

The tutorials cover these topics:

- Fundamental concepts of **machine learning** and their application to **electronic structure problems**. [Video](https://www.youtube.com/watch?v=V9wDgLjeJoE&t=1s)
- The inclusion of **nuclear quantum effects** using **path-integral molecular dynamics (PIMD)** via the i-PI code. [Video](https://www.youtube.com/live/5TakNe0Yn4s?si=RXdVgpIzREhmNhuJ&t=5138)
- Committee-based **active learning** techniques for interatomic potentials. [Video](https://www.youtube.com/live/f_u3txNm5wc?si=MwUiH9qFrJjYzrHU&t=7040)

---

## Tutorial 1: Machine Learning Basics and ML for Electronic Structure

### Run the Tutorial

It is recommended to use **Jupyter Lab**.

Switch to the repository root directory and run:
```bash
jupyter lab --notebook-dir=.
```

### Basics-GPR-and-NN Tutorials

In folder `Basics-GPR-and-NN`, you will find two self-contained Jupyter notebooks.  
They implement **Gaussian Process Regression (GPR)** and a **Neural Network (NN)** from scratch, fitting simple datasets for pedagogical purposes.

Install the required packages:

```bash
pip install numpy matplotlib plotly ipython ipykernel ipywidgets
```

### SALTED Tutorial

In folder `SALTED`, you will find a Jupyter notebook demonstrating the use of **SALTED** for learning the **electronic density of water monomers**.

**Requirements:**
- [SALTED](https://salted.readthedocs.io/en/latest/installation/)
- [FHI-aims](https://fhi-aims.org/get-the-code-menu/license-academia)

> ⚠️ *FHI-aims is only required for the extrapolation predictions at the end of the tutorial.*  

Please install the requirements:
1. Navigate to `SALTED-tutorial/tutorial-run` and decompress datasets:
   ```bash
   bash decompress_data.sh
   ```
2. Install the [SALTED Python package](https://github.com/andreagrisafi/SALTED)
3. Install required dependencies:
   ```bash
   pip install numpy matplotlib py3Dmol salted ase ipywidgets scipy scikit-learn h5py
   ```

---

## Tutorial 2: Nuclear Quantum Effects with i-PI

### Overview

The tutorial consists of two main parts:
1. **Equilibrium density of low-temperature para-hydrogen** via NpT simulation.
2. **Quantum free energy differences** using multiple computational strategies.


### Resources

- 🌐 [SabIA Research Group](https://www.mpsd.mpg.de/research/groups/sabia)
- 📖 [i-PI Documentation](https://docs.ipi-code.org/)


### Installation

Install the required Python packages (preferably in a virtual environment):

```bash
python -m pip install -r requirements.txt
```

Add the tutorial modules to your system path:
```bash
source env.sh
```

Clone and install **i-PI** (replace `/path/to/i-pi` with your desired location):
```bash
git clone git@github.com:i-pi/i-pi.git /path/to/i-pi
```

Compile the Fortran drivers:
```bash
cd /path/to/i-pi
cd drivers/f90
make
cd ../..
```

Add i-PI to your path:
```bash
source /path/to/i-pi/env.sh
```

You are now ready to start running the simulations.

---

## Tutorial 3: Machine Learning Interatomic Potentials – Active Learning

This tutorial illustrates the basic concepts of **committee-based active learning** using **MACE potentials** to represent potential energy surfaces in simple, illustrative systems.

You will find two tutorials in the `notebooks` folder:
- `1-zundel.ipynb`: apply Query by Committee to the *zundel cation* from labeled data
- `2-eigen.ipynb`: apply Query by Committee to the *eigen cation* from unlabeled data


### Resources

- ⚙️ [MACE Repository](https://github.com/ACEsuit/mace)
- 📦 [ASE Repository](https://gitlab.com/ase/ase)


### Installation

#### System Dependencies

This tutorial requires the system package `python3-tk` for GUI support (used by `ase` and related tools).  
Please install it before proceeding:

```bash
sudo apt-get install python3-tk
```

#### Option 1: Using `pip` (recommended for simplicity)

Create and activate a virtual environment and install the package and its dependencies through the `pyproject.toml` file:

```bash
mkdir -p ~/venv
python -m venv ~/venv/alt
source ~/venv/alt/bin/activate  
# On Windows use: %USERPROFILE%\venv\alt\Scripts\activate
python -m pip install -e .
```

#### Option 2: Using `conda` (alternative approach)

Create a `conda` environment using the provided `environment.yaml` file and activate the `alt` environment:

```bash
conda env create -f environment.yml
conda activate alt
```

#### Verify Installation

Check that the main packages we need are installed:

```bash
python tests/check.py
```

### References

The active learning are based on the following work:

- [Schran C., Brezina K., Marsalek O. JCP, 153, 104105, 2020](https://doi.org/10.1063/5.0016004)

---

## 5. Useful Links

- [SALTED Documentation](https://salted.readthedocs.io)
- [DeepH Repository](https://github.com/deeph-dev)
- [FHI-aims Website](https://fhi-aims.org)
- [i-PI Code Repository](https://github.com/i-pi/i-pi)
- [SabIA Group Webpage](https://www.mpsd.mpg.de/research/groups/sabia)
- [MACE Repository](https://github.com/ACEsuit/mace)
- [ASE Repository](https://gitlab.com/ase/ase)
- [Elia, Eliaaaaa!](https://www.youtube.com/clip/UgkxIIoxVQ6gTe0BnKlGReFWBo4DUQsdiSXa)