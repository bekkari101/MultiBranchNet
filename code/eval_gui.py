import io
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps, ImageDraw

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from hyperP import ModelConfig


class ParallelBranchNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.input_channels, config.conv_channels[0], kernel_size=config.conv_kernel_size, padding=config.conv_padding)
        self.pool1 = nn.MaxPool2d(kernel_size=config.pool_kernel_size, stride=config.pool_stride)
        self.conv2 = nn.Conv2d(config.conv_channels[0], config.conv_channels[1], kernel_size=config.conv_kernel_size, padding=config.conv_padding)
        self.pool2 = nn.MaxPool2d(kernel_size=config.pool_kernel_size, stride=config.pool_stride)

        self.branch_layers = nn.ModuleList([
            nn.ModuleList(
                [nn.Linear(config.input_branch_size, config.branch_size)] +
                [nn.Linear(config.branch_size, config.branch_size) for _ in range(config.branch_layers - 1)]
            )
            for _ in range(config.branches)
        ])

        self.fc_out = nn.Linear(config.hidden_size, config.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        pad = self.config.pad_size
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        chunks = torch.split(x, self.config.input_branch_size, dim=1)
        outs = []
        for i in range(self.config.branches):
            h = chunks[i]
            for linear in self.branch_layers[i]:
                h = self.relu(linear(h))
            outs.append(h)
        x = torch.cat(outs, dim=1)
        return self.fc_out(x)


def preprocess_pil_image(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert('L').resize((28, 28))
    img = ImageOps.invert(img)
    arr = np.array(img).astype(np.float32) / 255.0
    # Normalize using MNIST stats
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class DrawApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Eval GUI - Draw and Predict")

        self.canvas_size = 280
        self.brush_size = 12
        self.bg_color = 'black'
        self.fg_color = 'white'

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=8, pady=8)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-1>', self.draw)

        self.btn_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.btn_clear.grid(row=1, column=0, padx=4, pady=4, sticky='ew')

        self.btn_load = tk.Button(self.root, text="Select Checkpoint", command=self.select_checkpoint)
        self.btn_load.grid(row=1, column=1, padx=4, pady=4, sticky='ew')

        self.btn_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.btn_predict.grid(row=1, column=2, padx=4, pady=4, sticky='ew')

        self.status = tk.Label(self.root, text="No checkpoint selected", anchor='w')
        self.status.grid(row=2, column=0, columnspan=3, sticky='ew', padx=8, pady=4)

        # Internal buffers
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw_ctx = ImageDraw.Draw(self.image)

        # Model
        self.checkpoint_path: Path | None = None
        self.model: ParallelBranchNet | None = None
        self.model_config: ModelConfig | None = None

    def draw(self, event):
        x, y = event.x, event.y
        r = self.brush_size
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=self.fg_color, outline=self.fg_color)
        self.draw_ctx.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw_ctx = ImageDraw.Draw(self.image)

    def select_checkpoint(self):
        path = filedialog.askopenfilename(title="Select checkpoint (.pth)", filetypes=[("PyTorch checkpoint", "*.pth")])
        if not path:
            return
        self.checkpoint_path = Path(path)
        try:
            payload = torch.load(self.checkpoint_path, map_location='cpu')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint: {e}")
            return

        cfg_dict = payload.get("model_config") or payload.get("config")
        if cfg_dict is None:
            messagebox.showerror("Error", "Checkpoint missing model configuration.")
            return

        self.model_config = ModelConfig(
            conv_channels=tuple(cfg_dict.get("conv_channels", (12, 16))),
            branches=int(cfg_dict.get("branches", 6)),
            branch_size=int(cfg_dict.get("branch_size", 6)),
            branch_layers=int(cfg_dict.get("branch_layers", 2)),
        )
        self.model = ParallelBranchNet(self.model_config)

        state = payload.get("model_state_dict") or payload
        if not isinstance(state, dict):
            messagebox.showerror("Error", "Invalid checkpoint format: missing state dict.")
            return

        # Translate legacy keys (branch_fc1/branch_fc2[/branch_fc3]) to branch_layers.* format
        translated = self._maybe_translate_legacy_state(state, self.model_config)
        try:
            missing, unexpected = self.model.load_state_dict(translated, strict=False)
            if len(missing) > 0:
                print("Missing keys while loading (allowed):", missing)
            if len(unexpected) > 0:
                print("Unexpected keys while loading (ignored):", unexpected)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model state: {e}")
            return

        self.model.eval()
        self.status.config(text=f"Loaded: {self.checkpoint_path.name}")

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Model", "Please select a checkpoint first.")
            return

        img28 = self.image.resize((28, 28), Image.BILINEAR)
        tensor = preprocess_pil_image(img28)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0).detach().cpu().numpy()
            probs = softmax(logits)
            pred = int(np.argmax(probs))

        # Plot probabilities
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(list(range(10)), probs, color='steelblue')
        ax.set_xticks(range(10))
        ax.set_title(f"Prediction: {pred}")
        ax.set_ylim(0, 1)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=200)
        plt.close(fig)
        buf.seek(0)
        chart_img = Image.open(buf)

        # Show chart in a popup
        top = tk.Toplevel(self.root)
        top.title("Probabilities")
        from PIL import ImageTk
        tk_img = ImageTk.PhotoImage(chart_img)
        lbl = tk.Label(top, image=tk_img)
        lbl.image = tk_img  # keep reference
        lbl.pack()

    def run(self):
        self.root.mainloop()

    def _maybe_translate_legacy_state(self, state: Dict[str, Any], cfg: ModelConfig) -> Dict[str, Any]:
        # Detect legacy naming by presence of 'branch_fc1.0.weight'
        is_legacy = any(k.startswith('branch_fc1.') for k in state.keys()) or any(k.startswith('branch_fc2.') for k in state.keys())
        if not is_legacy:
            return state
        new_state = state.copy()
        # Map per-branch fully connected layers
        for i in range(cfg.branches):
            # fc1 -> branch_layers.i.0
            w1 = f'branch_fc1.{i}.weight'
            b1 = f'branch_fc1.{i}.bias'
            if w1 in state and b1 in state:
                new_state[f'branch_layers.{i}.0.weight'] = state[w1]
                new_state[f'branch_layers.{i}.0.bias'] = state[b1]
            # fc2 -> branch_layers.i.1 (if branch_layers >= 2)
            w2 = f'branch_fc2.{i}.weight'
            b2 = f'branch_fc2.{i}.bias'
            if cfg.branch_layers >= 2 and w2 in state and b2 in state:
                new_state[f'branch_layers.{i}.1.weight'] = state[w2]
                new_state[f'branch_layers.{i}.1.bias'] = state[b2]
            # Optional fc3 legacy -> branch_layers.i.2
            w3 = f'branch_fc3.{i}.weight'
            b3 = f'branch_fc3.{i}.bias'
            if cfg.branch_layers >= 3 and w3 in state and b3 in state:
                new_state[f'branch_layers.{i}.2.weight'] = state[w3]
                new_state[f'branch_layers.{i}.2.bias'] = state[b3]
        # Remove legacy keys to avoid unexpected
        for k in list(new_state.keys()):
            if k.startswith('branch_fc1.') or k.startswith('branch_fc2.') or k.startswith('branch_fc3.'):
                del new_state[k]
        return new_state


if __name__ == "__main__":
    app = DrawApp()
    app.run()


