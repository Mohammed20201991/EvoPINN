# EvoPINN: An Evolutionary Physics-Informed Neural Network Framework

EvoPINN is a hybrid machine learning framework that integrates **Genetic Algorithms (GA)** with **Physics-Informed Neural Networks (PINNs)** and traditional **Artificial Neural Networks (ANNs)** to model complex physical phenomena. It automates neural architecture and hyperparameter tuning using evolutionary strategies, while simultaneously embedding physical constraints into the learning process.

---

##  Overview

The EvoPINN framework is designed to:

- Leverage **Genetic Algorithms** for optimizing neural network structures (e.g., number of neurons, layers, learning rate).
- Incorporate **Physics-Informed Neural Networks** to enforce known physical laws through custom loss functions.
- Use **Artificial Neural Networks** to model relationships in datasets involving partial knowledge or observational data.
- Evaluate performance using standard regression metrics like **RMSE** and **R² Score**.

---

##  Features

- **Evolutionary Neural Architecture Search (ENAS)** for PINNs/ANNs
- Physics-based loss enforcement in training
- Scalable to both low- and high-dimensional regression problems
- Easy integration with PyTorch and DEAP

---

## 📁 Directory Structure

~~~
├── EvoPINN/
│ ├── models/ # GA+ANN and PINN model definitions
│ ├── ga_optimization/ 
│ ├── datasets/ 
│ ├── training/ 
│ ├── utils/ 
│ └── main.py 
~~~


---

## 🧪 Installation

```bash
pip install -r requirements.txt
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
openpyxl
deap
```

## Usage
`python main.py --data "datasets/data.xlsx" --generations 30 --population 20`

```
@misc{evopinn2025,
  title        = {EvoPINN: An Evolutionary Physics-Informed Neural Network Framework},
  author       = {Omar H. Jasim, Mohammed Al-Hitawi, Mohammed Y. Fattah, and Nameer A. Kareem},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/Mohammed20201991/EvoPINN}},
}
```
