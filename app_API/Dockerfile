# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app_API

# Copy the contents of the local app_API directory into the container at /app_API
COPY . /app_API

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask application
CMD ["python", "serve_model.py"]
