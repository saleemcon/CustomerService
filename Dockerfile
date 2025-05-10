FROM python:3.11-slim

# Install system dependencies (Tesseract, Poppler for PDF images)
RUN apt-get update && \
    apt-get install -y tesseract-ocr libgl1-mesa-glx poppler-utils && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port used by Streamlit
EXPOSE 8501

# Run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
