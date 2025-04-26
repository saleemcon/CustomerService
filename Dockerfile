# Use slim Python image
FROM python:3.11-slim

# Install system dependencies for OCR and PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8000

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
