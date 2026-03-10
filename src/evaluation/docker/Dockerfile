FROM mambaorg/micromamba:2.5.0

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# AiZynthFinder requires Python 3.11
RUN micromamba install --yes -c conda-forge \
    python=3.11 \
    aizynthfinder \
    && micromamba clean --all --yes

# FastAPI server
RUN pip install --no-cache-dir fastapi uvicorn

WORKDIR /app

COPY synthesizability.py /app/synthesizability.py
COPY docker/app.py /app/app.py

# Mount config YAML + model files at runtime.
# Usage: docker run -v /path/to/aizynthfinder-data:/data ...
# Expected layout:
#   /data/config.yml        — AiZynthFinder config YAML
#   /data/uspto_model.onnx  — expansion policy model
#   /data/stock.hdf5        — building blocks stock file
VOLUME ["/data"]

EXPOSE 8012
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012"]
