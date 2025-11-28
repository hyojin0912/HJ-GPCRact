# Base image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies, including bash
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev wget bash && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash /tmp/anaconda.sh -b -p /anaconda && \
    rm /tmp/anaconda.sh

# Set up Conda environment
ENV PATH="/anaconda/bin:/anaconda/envs/aigpro_env/bin:${PATH}"
COPY environment.yml /environment.yml
RUN conda env create -f /environment.yml -n aigpro_env && \
    conda clean --all -f -y

# Create the activate.d directory and set PYTHONPATH
RUN mkdir -p /anaconda/envs/aigpro_env/etc/conda/activate.d && \
    echo "export PYTHONPATH=/home/aigpro_api/aigpro:\$PYTHONPATH" >> /anaconda/envs/aigpro_env/etc/conda/activate.d/env_vars.sh && \
    chmod +x /anaconda/envs/aigpro_env/etc/conda/activate.d/env_vars.sh && \
    echo "export PYTHONPATH=/home/aigpro_api/aigpro:\$PYTHONPATH" >> /root/.bashrc

WORKDIR /home/aigpro_api

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install -r /requirements.txt

# Development image
FROM base AS development
# Additional development dependencies can be added here if needed

# Testing image
FROM base AS testing
RUN pip install pytest requests

# Production image
FROM base AS production
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev bash && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/aigpro_api
COPY . .

# Install additional dependencies (but NOT the aigpro package itself)
RUN pip install Werkzeug

# Ensure scripts are executable with absolute path and debug output
RUN chmod +x /home/aigpro_api/scripts/start.sh && \
    ls -l /home/aigpro_api/scripts/start.sh  # Verify permissions

# Environment variables for the application
ENV MODE=production
ENV APP_MODULE=aigpro.app:app
ENV WORKERS_PER_CORE=1
ENV HOST=0.0.0.0
ENV PORT=8097
ENV TZ="Asia/Seoul"
EXPOSE ${PORT}

# Create a wrapper script to initialize Conda environment and start a shell
RUN echo "#!/bin/bash\n\
set -e\n\
# Source the Conda environment directly\n\
. /anaconda/etc/profile.d/conda.sh && \
conda activate aigpro_env && \
echo 'Conda environment activated' && \
exec bash -i" > /entrypoint.sh && \
chmod +x /entrypoint.sh && \
ls -l /entrypoint.sh  # Verify permissions

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash", "-c", "while true; do sleep 3600; done"]  # Use JSON format to address warning