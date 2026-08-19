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
# Pin to a fixed AVX2+FMA target instead of NATIVE (avoids depending on the
# build machine's exact CPU) or full auto-detection (which was the extra
# compile cost that caused the OOM) -- AVX2/FMA are present on essentially
# every x86-64 server CPU Spaces runs on, so this keeps inference fast
# without the crash risk of a mismatched AVX-512 build.
COPY requirements.txt .
ENV CMAKE_BUILD_PARALLEL_LEVEL=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX512=OFF"
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
