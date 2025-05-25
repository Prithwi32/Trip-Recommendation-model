FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set up cache directory with correct permissions
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface
RUN mkdir -p /home/user/.cache/huggingface && \
    chown -R user:user /home/user/.cache

WORKDIR /code
RUN chown user:user /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Make sure the dataset is in the right place
COPY destinations_enhanced_global.csv .

# Switch to non-root user
USER user

# Expose the port that FastAPI will run on
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 