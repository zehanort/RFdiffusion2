# Dockerfile for RFdiffusion2 (adapted from RFdiffusion2 installation instructions)
# See: https://rosettacommons.github.io/RFdiffusion2/installation.html

FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive

# Basic system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and activate conda environment
RUN conda create -y -n rfd2 python=3.11 && \
    conda install -y -n rfd2 \
      -c nvidia/label/cuda-12.4.0 \
      -c pytorch \
      -c dglteam/label/th24_cu124 \
      -c conda-forge \
      pip \
      numpy"<2" \
      matplotlib \
      jupyterlab \
      openbabel==3.1.1 \
      cuda \
      pytorch==2.4 \
      pytorch-cuda==12.4

# Install pip packages in the rfd2 environment
RUN /bin/bash -c "source activate rfd2 && pip install \
  hydra-core==1.3.1 \
  ml-collections==0.1.1 \
  addict==2.4.0 \
  assertpy==1.1.0 \
  biopython==1.83 \
  colorlog \
  compact-json \
  cython==3.0.0 \
  cytoolz==0.12.3 \
  debugpy==1.8.5 \
  deepdiff==6.3.0 \
  dm-tree==0.1.8 \
  e3nn==0.5.1 \
  einops==0.7.0 \
  executing==2.0.0 \
  fastparquet==2024.5.0 \
  fire==0.6.0 \
  GPUtil==1.4.0 \
  icecream==2.1.3 \
  ipdb==0.13.11 \
  ipykernel==6.29.5 \
  ipython==8.27.0 \
  ipywidgets \
  mdtraj==1.10.0 \
  numba \
  omegaconf==2.3.0 \
  opt_einsum==3.3.0 \
  pandas==1.5.0 \
  plotly==5.16.1 \
  pre-commit==3.7.1 \
  py3Dmol==2.2.1 \
  pyarrow==17.0.0 \
  pydantic \
  pyrsistent==0.19.3 \
  pytest-benchmark \
  pytest-cov==4.1.0 \
  pytest-dotenv==0.5.2 \
  pytest==8.2.0 \
  rdkit==2024.3.5 \
  RestrictedPython \
  ruff==0.6.2 \
  scipy==1.13.1 \
  seaborn==0.13.2 \
  sympy==1.13.2 \
  tmtools \
  tqdm==4.65.0 \
  typer==0.12.5 \
  wandb==0.13.10"

# Install additional pip packages in the rfd2 environment
RUN /bin/bash -c "source activate rfd2 && \
    pip install biotite && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html && \
    pip install -U -i https://pypi.anaconda.org/rapidsai-wheels-nightly/simple 'pylibcugraphops-cu12>=24.6.0a24' && \
    pip install torchdata==0.9.0"

# Install dgl with the correct CUDA version in the rfd2 environment
RUN conda install -y -n rfd2 -c dglteam/label/th24_cu124 dgl

COPY . /opt/RFdiffusion2
ENV PYTHONPATH="/opt/RFdiffusion2"
ENV DGLBACKEND="pytorch"

# Set default environment
ENV CONDA_DEFAULT_ENV=rfd2
ENV PATH /opt/conda/envs/rfd2/bin:$PATH

# Ensure Conda is initialized for any shell
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate rfd2" >> ~/.bashrc

WORKDIR /opt/RFdiffusion2/rf_diffusion

# api setup
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate rfd2 && pip install fastapi[standard]"
COPY api /api

# Ensure uvicorn is launched from inside the conda environment.
# Use /bin/bash -lc in CMD to source conda and exec uvicorn.
CMD ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate rfd2 && exec uvicorn api.main:app --host 0.0.0.0 --port 8000"]
