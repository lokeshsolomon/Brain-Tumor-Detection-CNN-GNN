"""
model.py
========
Shared architecture definitions for the Brain Tumor CNN + GAT project.

This module is imported by:
  - brain_tumor_cnn_gnn.ipynb   (training / evaluation notebook also embeds
                                  a copy of this code so it can run standalone
                                  on Colab, but uses the same architecture)
  - app.py                      (Flask inference server)

Architecture summary
---------------------
1. CNNBackbone  : EfficientNet-B0 (pretrained) -> spatial feature map
2. GraphBuilder : feature map -> 7x7 patch graph (49 nodes, 4-connectivity)
3. GNNHead      : 3-layer Graph Attention Network (GAT) -> class logits
4. GradCAM      : Grad-CAM explainability hooked onto the CNN backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE    = 224
NODE_DIM    = 128
GNN_HIDDEN  = 256
GAT_HEADS   = 4
DROPOUT     = 0.4

# Class names MUST match the alphabetical order produced by
# torchvision.datasets.ImageFolder on the Training/ directory.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Human-readable descriptions shown in the web UI
CLASS_INFO = {
    "glioma": {
        "label": "Glioma",
        "tumor_present": True,
        "description": "A tumor that arises from glial cells in the brain or spine.",
    },
    "meningioma": {
        "label": "Meningioma",
        "tumor_present": True,
        "description": "A tumor that forms on membranes covering the brain and spinal cord.",
    },
    "notumor": {
        "label": "No Tumor",
        "tumor_present": False,
        "description": "No tumor was detected in the scan.",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "tumor_present": True,
        "description": "A tumor that develops in the pituitary gland.",
    },
}


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """Return torchvision transforms for 'train' or 'val'/'test' splits."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                    saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                     scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 1. CNN Backbone (EfficientNet-B0, pretrained)
# ---------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    """
    EfficientNet-B0 with the classifier head removed.
    Returns a spatial feature map: (B, C, H', W')

    The number of output channels is detected dynamically with a dummy
    forward pass, so this class works unchanged if you swap in a different
    timm backbone (e.g. 'resnet50', 'efficientnet_b3').
    """

    def __init__(self, model_name: str = "efficientnet_b0",
                 freeze_until_layer: int = 5, pretrained: bool = True):
        super().__init__()
        base = timm.create_model(model_name, pretrained=pretrained,
                                  features_only=True)
        self.features = nn.Sequential(*list(base.children())[:freeze_until_layer])

        # Freeze the earliest layers for stable fine-tuning
        for i, child in enumerate(self.features.children()):
            if i < 3:
                for p in child.parameters():
                    p.requires_grad = False

        # Dynamically determine output channel count
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            out = self.features(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            self.out_channels = out.shape[1]

    def forward(self, x):
        out = self.features(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out                       # (B, C, H', W')


# ---------------------------------------------------------------------------
# 2. Graph Construction
# ---------------------------------------------------------------------------
class GraphBuilder(nn.Module):
    """
    Converts CNN feature map patches into graph nodes.
    Each patch -> one node; edges connect spatially adjacent patches
    on a 7x7 grid (4-connectivity, 49 nodes total).
    """

    def __init__(self, in_channels: int, node_dim: int = NODE_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),    # fixed 7x7 = 49 patches
            nn.Flatten(start_dim=2),         # keep batch & channel dims
        )
        self.node_embed = nn.Linear(in_channels, node_dim)
        self._build_static_edges()

    def _build_static_edges(self):
        G = 7
        src, dst = [], []
        for r in range(G):
            for c in range(G):
                n = r * G + c
                if c + 1 < G:
                    src += [n, n + 1]
                    dst += [n + 1, n]
                if r + 1 < G:
                    src += [n, n + G]
                    dst += [n + G, n]
        self.register_buffer("edge_index",
                              torch.tensor([src, dst], dtype=torch.long))

    def forward(self, feat_map):
        """
        feat_map: (B, C, H, W)
        Returns a list of torch_geometric.data.Data objects (one per sample)
        """
        B, C, _, _ = feat_map.shape
        x = self.proj(feat_map).permute(0, 2, 1)   # (B, 49, C)
        x = self.node_embed(x)                     # (B, 49, node_dim)

        graphs = []
        for i in range(B):
            graphs.append(Data(x=x[i], edge_index=self.edge_index.clone()))
        return graphs


# ---------------------------------------------------------------------------
# 3. GNN Head (3-layer Graph Attention Network)
# ---------------------------------------------------------------------------
class GNNHead(nn.Module):
    """
    3-layer Graph Attention Network (GAT).
    Global mean pooling -> per-graph embedding -> classifier.
    """

    def __init__(self, node_dim: int = NODE_DIM, hidden: int = GNN_HIDDEN,
                 num_classes: int = NUM_CLASSES, dropout: float = DROPOUT,
                 heads: int = GAT_HEADS):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden * heads, hidden // 2, heads=1,
                              concat=False, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei)); x = self.drop(x)
        x = F.elu(self.conv2(x, ei)); x = self.drop(x)
        x = F.elu(self.conv3(x, ei))
        x = global_mean_pool(x, batch)        # (B, hidden//2)
        return self.cls(x)


# ---------------------------------------------------------------------------
# 4. Full Hybrid Model
# ---------------------------------------------------------------------------
class BrainTumorDetector(nn.Module):
    def __init__(self, cnn_name: str = "efficientnet_b0",
                 node_dim: int = NODE_DIM, gnn_hidden: int = GNN_HIDDEN,
                 dropout: float = DROPOUT, pretrained: bool = True):
        super().__init__()
        self.cnn   = CNNBackbone(cnn_name, pretrained=pretrained)
        self.graph = GraphBuilder(self.cnn.out_channels, node_dim)
        self.gnn   = GNNHead(node_dim, gnn_hidden, NUM_CLASSES, dropout)

    def forward(self, images):
        feat_map = self.cnn(images)              # (B, C, H, W)
        graphs   = self.graph(feat_map)
        batch    = Batch.from_data_list(graphs).to(images.device)
        return self.gnn(batch)


# ---------------------------------------------------------------------------
# 5. Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM for the hybrid model, hooked onto the last layer of the CNN
    backbone (the feature map that feeds the graph builder).

    Usage:
        cam_tool = GradCAM(model)
        heatmap, pred_class, probs = cam_tool(input_tensor)
    """

    def __init__(self, model: BrainTumorDetector):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.cnn.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: (1, 3, H, W)
        Returns: (heatmap [H,W] in [0,1], predicted/used class index, probs)
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(probs.argmax().item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None:
            raise RuntimeError("GradCAM failed to capture activations during forward pass.")
        if self.gradients is None:
            raise RuntimeError("GradCAM failed to capture gradients during backward pass.")

        gradients = self.gradients[0]            # (C, h, w)
        activations = self.activations[0]        # (C, h, w)

        weights = gradients.mean(dim=(1, 2))      # (C,)
        cam = F.relu((weights[:, None, None] * activations).sum(0))

        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        return cam.cpu().numpy(), class_idx, probs.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Helper: load a trained checkpoint
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device,
                cnn_name: str = "efficientnet_b0") -> BrainTumorDetector:
    """Instantiate the model architecture and load trained weights."""
    model = BrainTumorDetector(cnn_name=cnn_name, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
