# 🌿 Classify Waste Products Using Transfer Learning

> **IBM AI Engineering Professional Certificate — Module 7 Final Project**  
> VGG16 Transfer Learning & Fine-Tuning for Organic vs Recyclable Waste Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Project Overview

This project builds a **binary image classifier** that distinguishes between **Organic (O)** and **Recyclable (R)** waste using **VGG16 Transfer Learning** pre-trained on ImageNet.

Two complete submission apps are provided, each as a self-contained Streamlit application:

| | Option 1 — AI Graded | Option 2 — Peer Reviewed |
|---|---|---|
| **Folder** | `Option 1 - AI Graded/` | `Option 2 - Peer Reviewed/` |
| **Port** | `8501` | `8502` |
| **Theme** | Cyan / Dark | Green / Dark |
| **Extra Features** | Automated pipeline | Live Classifier page |

---

## 🏗️ Architecture

```
VGG16 (ImageNet, frozen) → Flatten → Dense(512, ReLU) → Dropout(0.3)
                                   → Dense(512, ReLU) → Dropout(0.3)
                                   → Dense(1, Sigmoid)
```

**Training Strategy:**
- **Phase 1 — Extract Features:** VGG16 weights frozen, only new head trained
- **Phase 2 — Fine-Tuning:** Top layers of VGG16 unfrozen (`block5_conv3` onwards), lower learning rate

---

## 📁 Repository Structure

```
├── README.md
├── Final Proj-Classify Waste Products Using TL FT.md   ← Project brief
│
├── Option 1 - AI Graded/
│   ├── app.py              ← Streamlit app (AI-Graded submission)
│   └── requirements.txt
│
└── Option 2 - Peer Reviewed/
    ├── app.py              ← Streamlit app (Peer-Review submission)
    └── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### 1. Clone the repo
```bash
git clone https://github.com/aiprojects2060/Classify-Waste-Products-Using-Transfer-Learning.git
cd Classify-Waste-Products-Using-Transfer-Learning
```

### 2. Run Option 1 — AI Graded
```bash
cd "Option 1 - AI Graded"
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```
Open → **http://localhost:8501**

### 3. Run Option 2 — Peer Reviewed
```bash
cd "Option 2 - Peer Reviewed"
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```
Open → **http://localhost:8502**

---

## 📊 App Walkthrough

### Option 1 — AI Graded (`localhost:8501`)

| Step | What happens |
|------|-------------|
| Open app | Navigate to **Tasks 1–5** in the sidebar |
| Auto-setup | Dataset downloads automatically, data generators build, VGG16 model summary displays, model compiles — all without any clicks |
| Train | Click **▶ Start Training Now** — both Phase 1 and Phase 2 train back-to-back |
| Stop | **⏹ Stop Training** button is live throughout — stops cleanly at end of current epoch |
| Curves | Navigate to **Tasks 6–8** for training/validation accuracy & loss curves |
| Predictions | Navigate to **Tasks 9–10** for test image predictions from both models |

### Option 2 — Peer Reviewed (`localhost:8502`)

Same flow as Option 1 plus an additional:

| Extra Page | Feature |
|-----------|---------|
| 🔬 Live Classifier | Upload any waste image → instant prediction with confidence % |

---

## ✅ Tasks Covered

| Task | Description |
|------|-------------|
| Task 1 | Print TensorFlow version |
| Task 2 | Create `test_generator` |
| Task 3 | Print `len(train_generator)` |
| Task 4 | Print model `summary()` |
| Task 5 | Compile the model |
| Task 6 | Plot accuracy curves — Extract Features model |
| Task 7 | Plot loss curves — Fine-Tuned model |
| Task 8 | Plot accuracy curves — Fine-Tuned model |
| Task 9 | Predict test image (index=1) — Extract Features model |
| Task 10 | Predict test image (index=1) — Fine-Tuned model |

---

## 🔧 Dependencies

```txt
tensorflow>=2.12
streamlit>=1.33
numpy
Pillow
scikit-learn
matplotlib
requests
```

Install all at once:
```bash
pip install tensorflow streamlit numpy Pillow scikit-learn matplotlib requests
```

---

## 📦 Dataset

The dataset is **automatically downloaded** on first launch from IBM Cloud Object Storage:

```
Source: IBM Skills Network
Classes: Organic (O) · Recyclable (R)
Split:   train/ (80%) · test/ (20%)
Size:    ~1,200 images (reduced set)
```

No manual download required — the app handles it.

---

## 🏋️ Training Details

| Parameter | Value |
|-----------|-------|
| Image size | 150 × 150 |
| Batch size | 32 |
| Steps per epoch | 5 |
| Max epochs | 10 (each phase) |
| Phase 1 optimizer | Adam (lr=1e-5) |
| Phase 2 optimizer | RMSprop (lr=1e-4) |
| Early stopping | patience=4, min_delta=0.01 |
| LR schedule | Exponential decay |

---

## 🎨 UI Features

- 🌑 Premium dark theme with gradient backgrounds
- 📊 Real-time training progress (epoch, accuracy, val_accuracy)
- ⏹ Live Stop Training button (stops at epoch boundary)
- 📈 Publication-quality matplotlib training curves
- 🔬 Interactive live image classifier (Option 2)
- 🎉 Balloon animation on training completion

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using TensorFlow, VGG16, and Streamlit*
