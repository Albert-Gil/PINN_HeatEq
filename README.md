# 🔥 Physics-Informed Neural Networks for the Heat Equation

This repository contains two Python scripts that solve the unsteady heat equation in 2D and 3D using **Physics-Informed Neural Networks (PINNs)**, implemented with [DeepXDE](https://github.com/lululxvi/deepxde).

## 📄 Files

- `2D_HeatEq.py` — Solves the 2D heat equation on a unit square.
- `3D_HeatEqCube.py` — Solves the 3D heat equation on a unit cube.

## 🧩 Problem Description

### 🔷 2D Heat Equation

\[
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
\]

**Initial condition:**

\[
u(x, y, 0) = \sin(\pi x) \sin(\pi y)
\]

**Boundary condition (Dirichlet):**

\[
u(x, y, t) = 0 \quad \text{for all } (x, y) \text{ on the boundary}
\]

---

### 🔷 3D Heat Equation

\[
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)
\]

**Initial condition:**

\[
u(x, y, z, 0) = \sin(\pi x) \sin(\pi y) \sin(\pi z)
\]

**Boundary condition (Dirichlet):**

\[
u(x, y, z, t) = 0 \quad \text{for all } (x, y, z) \text{ on the boundary}
\]

## 🧠 Method

- The problem is defined on a spatial domain of `[0, 1]` in each dimension, and a time domain from `t = 0` to `t = T`.
- A feedforward fully-connected neural network is trained using a combination of Adam and L-BFGS optimizers.
- The model is trained to satisfy both the PDE and the boundary/initial conditions.

## 📽️ Visualization

Both scripts include:
- Comparison plots between PINN prediction and analytical solution
- Animations over time
- L2 relative error plots
- Snapshots at selected time instants

## ⚙️ Requirements

To run the code, you need:
- Python (3.7+)
- [`deepxde`](https://github.com/lululxvi/deepxde)
- `numpy`
- `matplotlib`
- FFmpeg installed and in your system path (for saving animations)

## 👨‍🔬 Author

**Albert Gil**  
PhD student at UC Irvine  
Focus: Liquid Hydrogen Systems and Machine Learning  
[LinkedIn](https://www.linkedin.com/in/albert-gil)
