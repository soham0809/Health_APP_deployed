# Use Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y libgl1
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask runs on 5000 by default)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
