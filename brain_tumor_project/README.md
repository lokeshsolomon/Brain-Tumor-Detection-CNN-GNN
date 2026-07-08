# Brain Tumor Detection & Classification — CNN + GAT

A hybrid deep-learning project that classifies brain MRI scans using a
**Convolutional Neural Network (CNN)** for feature extraction combined with a
**Graph Attention Network (GAT)** for relational reasoning over image
patches, with **Grad-CAM** explainability and a **Flask** web interface for
interactive predictions.

---

## 1. Overview

| Component | Choice |
|---|---|
| CNN backbone | EfficientNet-B0 (pretrained on ImageNet) |
| Graph construction | 7×7 patch grid (49 nodes), 4-connected edges |
| GNN head | 3-layer Graph Attention Network (GAT) |
| Explainability | Grad-CAM on the CNN backbone's last feature map |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Metrics | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Deployment | Flask web app with image upload + Grad-CAM overlay |

### A note on "tumor stage" classification

Public brain-MRI datasets (e.g. the Kaggle *Brain Tumor MRI Dataset*) are
labeled by **tumor type** — glioma, meningioma, pituitary, no-tumor — not by
clinical stage (Stage I–IV). Clinical staging requires expert
radiological/pathological annotations that aren't present in these public
datasets.

This project therefore classifies **tumor presence and tumor type**, which
is the standard formulation in the literature and what the data supports. The
code is fully generic: if you have a dataset labeled by stage instead, just
point `DATA_ROOT` at a folder with the same `Training/<class>/...` and
`Testing/<class>/...` structure (where `<class>` are your stage labels) and
update `CLASS_NAMES` in `model.py` — no other changes are needed.

---

## 2. Project Structure

```
brain_tumor_project/
├── brain_tumor_cnn_gnn.ipynb   # Training & evaluation notebook
├── model.py                    # Shared model architecture (CNN+GAT, Grad-CAM)
├── app.py                      # Flask backend (inference server)
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html              # Upload page
│   └── result.html             # Prediction result page
├── static/
│   ├── style.css                # Shared stylesheet
│   └── uploads/                 # Uploaded images + Grad-CAM overlays (created at runtime)
└── models/
    └── best_model.pth           # Trained checkpoint (created by the notebook)
```

---

## 3. Dataset

This project is designed for the **Brain Tumor MRI Dataset** (Masoud
Nickparvar, Kaggle), or any dataset with the same layout:

```
DATA_ROOT/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Download it from Kaggle and extract it so that `Training/` and `Testing/`
sit inside a single `DATA_ROOT` folder.

---

## 4. Setup

### 4.1 Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note on `torch-geometric`**: depending on your OS/CUDA version, you may
> need to follow the official install instructions at
> https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
> if `pip install torch-geometric` doesn't pick up the right wheel for your
> PyTorch + CUDA combination.

### 4.2 (Optional) Google Colab

The notebook includes an optional first cell for mounting Google Drive and
installing `timm` / `torch-geometric` on Colab — uncomment it if you're
running there.

---

## 5. Training the Model

1. Open `brain_tumor_cnn_gnn.ipynb` in Jupyter / Colab.
2. Run all cells in order. The notebook will:
   - Load and augment the dataset (resize, flips, rotation, color jitter,
     affine transforms, normalization).
   - Build the CNN (EfficientNet-B0) + GAT hybrid model.
   - Train for `EPOCHS` epochs with label smoothing, AdamW (differential
     learning rates for backbone vs. head), cosine annealing, gradient
     clipping, and mixed-precision (AMP).
   - Evaluate on the test set and print **accuracy, per-class precision /
     recall / F1-score**, and a full classification report.
   - Plot the **training/validation loss and accuracy curves** and the
     **confusion matrix**.
   - Run **Grad-CAM** on a sample image to visualize the regions that drove
     the prediction.
3. Set `DATA_ROOT` (in the "Run Training" cell) to your dataset path.
4. The best checkpoint is saved to `models/best_model.pth` — this is the
   file the Flask app loads.

---

## 6. Running the Web App

Once `models/best_model.pth` exists:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

1. Drag and drop (or browse for) an MRI image on the home page.
2. Click **Analyze Scan**.
3. The result page shows:
   - **Tumor presence** (detected / not detected)
   - **Predicted tumor type** (glioma / meningioma / pituitary / no tumor)
   - **Confidence score** for the top prediction
   - **Per-class probability bars** for all four classes
   - A toggle between the **original image** and the **Grad-CAM heatmap
     overlay**, highlighting the regions the model focused on

If `models/best_model.pth` does not exist yet, the app will still start (so
you can preview the UI), but predictions will come from randomly-initialized
weights — train the model first for meaningful results.

---

## 7. Architecture Details

### CNN Backbone (`CNNBackbone`)
EfficientNet-B0 (via `timm`, `features_only=True`), with the earliest layers
frozen for stable fine-tuning. Output channel count is detected dynamically,
so you can swap in `resnet50`, `efficientnet_b3`, etc. by changing
`cnn_name` in `BrainTumorDetector`.

### Graph Construction (`GraphBuilder`)
The CNN feature map is adaptively pooled to a 7×7 grid (49 patches). Each
patch is projected to a `node_dim`-dimensional embedding and becomes a graph
node. Edges connect 4-connected neighbors on the grid (same structure for
every image, precomputed once).

### GNN Head (`GNNHead`)
A 3-layer **Graph Attention Network (GAT)**:
- Layer 1: `GATConv(node_dim → hidden)`, multi-head attention
- Layer 2: `GATConv(hidden×heads → hidden)`, multi-head attention
- Layer 3: `GATConv(hidden×heads → hidden/2)`, single head
- Global mean pooling → MLP classifier head

GAT was chosen over a plain GCN so the model learns *attention weights* over
neighboring patches — letting it focus more on the patches most relevant to
the diagnosis.

### Grad-CAM (`GradCAM`)
Hooks are registered on the last layer of the CNN backbone to capture
activations and gradients. A forward + backward pass through the *entire*
hybrid model (CNN → graph → GAT) produces a class-discriminative heatmap,
which is resized and overlaid on the original image.

---

## 8. Metrics

`print_metrics()` in the notebook reports:
- Overall accuracy
- Per-class precision, recall, F1-score, and support
- Macro and weighted averages (via `sklearn.metrics.classification_report`)

`plot_confusion()` renders a confusion matrix heatmap.

---

## 9. Extending This Project

- **Different backbone**: pass `cnn_name="resnet50"` (or any `timm` model
  name) to `BrainTumorDetector`.
- **GCN instead of GAT**: swap `GATConv` for `GCNConv` in `GNNHead` (the
  rest of the pipeline is unchanged).
- **More/different classes (e.g. tumor stages)**: update `CLASS_NAMES` and
  `CLASS_INFO` in `model.py`, and point `DATA_ROOT` at a dataset with the
  matching `Training/<class>/` and `Testing/<class>/` folders.
- **Finer patch graphs**: change the `(7, 7)` grid size in `GraphBuilder`
  (and update `_build_static_edges` accordingly) for more/fewer nodes.

---

## 10. Disclaimer

This project is for **educational and research purposes only**. It is not a
medical device and must not be used for real clinical diagnosis or treatment
decisions.
