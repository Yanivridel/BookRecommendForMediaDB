# Use a lightweight Python image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Set the command to run your application
CMD ["python", "booksML.py"]