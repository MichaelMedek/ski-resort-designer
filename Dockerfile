# Hugging Face Docker Space for the Ski Resort Planner.
# Server flags live in .streamlit/config.toml. The 285 MB DEM is baked from the HF dataset at
# build time, so cold boots don't re-download it.

FROM python:3.11-slim

# HF Spaces run as UID 1000; create the user before any COPY.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install deps first (layer-cached). requirements.txt ends with `-e .`, whose setup.py reads
# version.txt and README.md at build time, so copy those too.
COPY --chown=user requirements.txt pyproject.toml setup.py version.txt README.md ./
COPY --chown=user skiresort_planner ./skiresort_planner
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the DEM so DEMConfig.EURODEM_PATH (data/alps_dem.tif) resolves with no runtime download.
ADD --chown=user \
    https://huggingface.co/datasets/MichaelMedek/alps_eurodem/resolve/main/alps_dem.tif \
    data/alps_dem.tif

EXPOSE 7860

CMD ["streamlit", "run", "skiresort_planner/app.py"]
