"""Layer 1 — Convolutional Autoencoder (Gatekeeper).

Responsibilities
----------------
- Train a convolutional autoencoder on defect-free images only.
- At inference time, compute per-image MSE reconstruction error.
- If MSE > threshold → pass frame to Layer 2 via Service Bus.
- If MSE ≤ threshold → discard frame (no defect detected).
"""
