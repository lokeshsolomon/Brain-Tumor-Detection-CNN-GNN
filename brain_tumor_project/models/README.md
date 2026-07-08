# models/

This directory holds the trained model checkpoint used by `app.py`.

Run `brain_tumor_cnn_gnn.ipynb` to train the model — it will save
`best_model.pth` here automatically (see the "Run Training" cell, where
`SAVE_PATH = "models/best_model.pth"`).

The checkpoint is a dict with:
- `model_state_dict` — the trained `BrainTumorDetector` weights
- `epoch` — epoch at which this checkpoint was saved
- `val_acc` — validation accuracy at that epoch
- `class_names` — the class label order used during training

`app.py` will start without this file (for UI preview), but predictions will
be meaningless until a real `best_model.pth` is placed here.
