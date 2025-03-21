# Use the official Python 3.11 full image as a base
FROM python:3.11

# Install system dependencies (optional, but can still be added if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.address", "0.0.0.0", "--server.fileWatcherType", "none"]


