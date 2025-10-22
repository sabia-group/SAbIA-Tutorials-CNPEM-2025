# CNPEM/Ilum INCT - Max Planck Meeting Tutorials 2025

## 🗂️ Table of Contents

1. [Overview](#1-overview)
2. [Tutorial 1: Machine Learning Basics and ML for Electronic Structure](#2-tutorial-1-machine-learning-basics-and-ml-for-electronic-structure)
   - [Run the Tutorial](#run-the-tutorial)
   - [Basics-GPR-and-NN](#basics-gprandnn-tutorials)
   - [SALTED](#salted-tutorial)
3. [Tutorial 2: Nuclear Quantum Effects with i-PI](#3-tutorial-2-nuclear-quantum-effects-with-i-pi)
   - [Overview](#overview-1)
   - [Resources](#resources)
   - [Installation](#installation)
4. [References and Links](#4-references-and-links)

---

## 1. Overview

Collection of tutorials from the **SabIA group** presented at the **2025 INCT - Max Planck Meeting on Electronic Structure Methods and Materials Informatics** at **CNPEM (Campinas, Brazil)**.

The tutorials cover these topics:

- Fundamental concepts of **machine learning** and their application to **electronic structure problems**.
- The inclusion of **nuclear quantum effects** using **path-integral molecular dynamics (PIMD)** via the i-PI code.
- ...

---

## 2. Tutorial 1: Machine Learning Basics and ML for Electronic Structure

### Run the Tutorial

It is recommended to use **Jupyter Lab**.

Switch to the repository root directory and run:
```bash
jupyter lab --notebook-dir=.
```

---

### Basics-GPR-and-NN Tutorials

In folder `Basics-GPR-and-NN`, you will find two self-contained Jupyter notebooks.  
They implement **Gaussian Process Regression (GPR)** and a **Neural Network (NN)** from scratch, fitting simple datasets for pedagogical purposes.

Install the required packages:

```bash
pip install numpy matplotlib plotly ipython ipykernel ipywidgets
```

---

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

## 3. Tutorial 2: Nuclear Quantum Effects with i-PI

### Overview

The tutorial consists of two main parts:
1. **Equilibrium density of low-temperature para-hydrogen** via NpT simulation.
2. **Quantum free energy differences** using multiple computational strategies.

---

### Resources

- 🌐 [SabIA Research Group](https://www.mpsd.mpg.de/research/groups/sabia)
- 📖 [i-PI Documentation](https://docs.ipi-code.org/)

---

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

## 4. References and Links

- [SALTED Documentation](https://salted.readthedocs.io)
- [DeepH Repository](https://github.com/deeph-dev)
- [FHI-aims Website](https://fhi-aims.org)
- [i-PI Code Repository](https://github.com/i-pi/i-pi)
- [SabIA Group Webpage](https://www.mpsd.mpg.de/research/groups/sabia)