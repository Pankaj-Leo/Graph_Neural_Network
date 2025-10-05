# 🧠AI Projects Suite
*A collection of graph-based, text-based, and spatiotemporal machine learning applications.*

---

## 🌍 Overview

This repository brings together three advanced AI projects that demonstrate deep learning across **graphs**, **text**, and **time series** domains:

| Domain | Project | Description |
|--------|----------|--------------|
| 🧬 Graph ML | **Molecular Property Prediction using Graph Neural Networks** (`gnn-project-main/`) | Predicts molecular behavior from SMILES representations using PyTorch Geometric GNNs. |
| 📰 NLP | **Fake News Detection** (`fake_news_detection.ipynb`) | Classifies true vs fake news using embeddings and transformer-based models. |
| 🚦 Time Series | **Traffic Flow Prediction** (`traffic_prediction.ipynb`) | Predicts future vehicle counts using LSTMs and temporal feature engineering. |

Each project is modular, self-contained, and aligned with modern ML best practices (reproducibility, explainability, and experiment tracking via MLflow).

---

## 🧩 Project 1 — Graph Neural Network (GNN) for Molecule Prediction

### 📁 Folder: `gnn-project-main/`

#### 🔍 Objective
Predict binary molecular properties (e.g., toxicity, activity) using molecular graph representations. Each molecule is parsed from SMILES strings and converted into nodes (atoms) and edges (bonds).

#### 🧱 Tech Stack
- **PyTorch Geometric (PyG)** for graph modeling  
- **RDKit** for molecular graph parsing  
- **MLflow** for experiment tracking  
- **Mango / Bayesian Optimization** for hyperparameter tuning

#### ⚙️ Core Files
| File | Purpose |
|------|----------|
| `dataset.py` | Converts SMILES to molecular graphs, handles RDKit featurization |
| `model.py` | Defines GNN model (message passing, pooling, MLP classifier) |
| `train.py` | Training loop, metric logging, MLflow tracking |
| `config.py` | Stores hyperparameter search space |
| `requirements.txt` | Environment dependencies |

#### 🧠 Key Features
- Custom PyTorch Geometric dataset class (`MoleculeDataset`)  
- Robust RDKit preprocessing with invalid SMILES handling  
- Early stopping & precision–recall balancing  
- Automated experiment logging via MLflow  
- Bayesian hyperparameter optimization (Mango Tuner)

#### 🚀 Example Run
```bash
python train.py
```

MLflow UI for metrics:
```bash
mlflow ui --port 5000
```

---

## 📰 Project 2 — Fake News Detection (NLP)

### 📁 File: `fake_news_detection.ipynb`

#### 🔍 Objective
Detect fake or misleading news articles using modern NLP pipelines and transformer embeddings.

#### 🧱 Techniques
- Text cleaning & tokenization (NLTK, spaCy)
- TF-IDF & word embedding feature extraction
- Classifiers: Logistic Regression, LSTM, BERT
- Model evaluation: F1, ROC-AUC, confusion matrix

#### ⚙️ Key Libraries
`pandas`, `scikit-learn`, `torch`, `transformers`, `nltk`

#### 📊 Outputs
- Classification report
- Word cloud visualizations
- ROC curve & confusion matrix plots

#### 🚀 Quick Start
```bash
jupyter notebook fake_news_detection.ipynb
```

---

## 🚦 Project 3 — Traffic Prediction (Time Series)

### 📁 File: `traffic_prediction.ipynb`

#### 🔍 Objective
Forecast traffic flow (vehicle count) using historical data and time series modeling.

#### 🧱 Techniques
- Data resampling & seasonal decomposition  
- Multivariate LSTM & GRU models  
- Sliding window sequence generation  
- Evaluation: RMSE, MAE, R²

#### ⚙️ Key Libraries
`pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow` or `torch`, `sklearn`

#### 📈 Workflow
1. Load and visualize traffic data  
2. Normalize and window sequences  
3. Train LSTM on historical data  
4. Predict and plot against actual flow  

---

## 🧪 Environment Setup

Create and activate a virtual environment:

```bash
conda create -n deepfindr python=3.10 -y
conda activate deepfindr
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Logging & Monitoring

All experiments are tracked using **MLflow**:
- Metrics: loss, accuracy, ROC-AUC  
- Artifacts: confusion matrices, trained models  
- Model registry: automatic versioning

Launch dashboard:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 📈 Results Snapshot

| Project | Model | Key Metric |
|----------|--------|-------------|
| GNN Molecule | GraphConv + GlobalMeanPool | ROC-AUC ≈ 0.89 |
| Fake News | BERT | F1 ≈ 0.92 |
| Traffic Forecasting | LSTM | RMSE ≈ 15.3 |

---

## 🧰 Requirements
```txt
torch
torch-geometric
rdkit
pandas
numpy
scikit-learn
matplotlib
mlflow
mango
tqdm
transformers
tensorflow
seaborn
```

---

## 📘 Citation
If you use this repository or its models for academic or professional work, please cite:

```
@misc{deepfindr2025gnn,

  url          = {https://github.com/deepfindr}
}
```

---

## 🧑‍💻 Author
**Pankaj Somkuwar**  
🔗 [GitHub](https://github.com/Pankaj-Leo) | [LinkedIn](https://linkedin.com/in/pankajsomkuwar)

---

## 🏁 License
This repository is released under the **MIT License** — free for academic and commercial use.
