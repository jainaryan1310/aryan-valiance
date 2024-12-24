# Use the official Python 3.11 image as the base
FROM python:3.11-slim

ENV GOOGLE_APPLICATION_CREDENTIALS=service.json

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI
CMD ["fastapi", "dev", "api.py", "--host", "0.0.0.0", "--port", "8000"]