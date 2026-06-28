FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# HF Spaces require a non-root uid 1000 with a writable home (model cache lands here)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GIT_LFS_SKIP_SMUDGE=1 PYTHONUNBUFFERED=1

WORKDIR /home/user/app
COPY --chown=user app.py .
# torch + CUDA come from the base image; only add timesfm + the web server.
RUN pip install --no-cache-dir \
    "timesfm @ git+https://github.com/google-research/timesfm.git" \
    fastapi "uvicorn[standard]"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
