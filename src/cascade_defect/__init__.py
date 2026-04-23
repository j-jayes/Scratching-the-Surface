"""
Cascade Defect Detection
========================
Three-layer cascade ML pipeline for real-time rolled-metal defect detection.

Layers
------
Layer 1 — Gatekeeper : Convolutional Autoencoder (anomaly filter via MSE)
Layer 2 — Specialist  : YOLOv8 (defect classification + localisation)
Layer 3 — Oracle      : GPT-4o (edge-case reasoning via few-shot prompting)
"""

__version__ = "0.1.0"
