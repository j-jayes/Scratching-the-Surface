FROM cascadedevacr6ya7a3.azurecr.io/cascade-base:latest
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN /usr/local/bin/uv pip install --python /app/.venv/bin/python --reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision
RUN /app/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.version.cuda)"
