# Use official Python image (with pip installed)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements first (for cache)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the Flask port
ENV PORT=5000

# Run the Flask app
CMD ["python", "app.py"]