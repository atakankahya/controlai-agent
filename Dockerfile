FROM python:3.11-slim

# Prevent interactive prompts & buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOME=/home/user

# Install essential scientific computation libraries and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces security standard
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python dependencies. llama-cpp-python compiles from source (no
# prebuilt wheel is published for recent versions); cap build parallelism so
# the compiler doesn't spawn enough jobs to OOM-kill the Spaces build machine.
COPY requirements.txt .
ENV CMAKE_BUILD_PARALLEL_LEVEL=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Ensure storage directories exist with write access
RUN mkdir -p outputs/plots data/rag_index data/user_docs && \
    chown -R user:user /home/user

USER user

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Launch FastAPI web console
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
