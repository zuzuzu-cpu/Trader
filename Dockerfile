# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables (these should also be provided at runtime via .env or docker-compose)
ENV PYTHONUNBUFFERED=1

# Command to run the application
# We use a simple loop in main.py or could use a cron job/scheduler here
CMD ["python", "main.py"]
