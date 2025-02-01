# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure /tmp is writable
RUN chmod -R 777 /tmp

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8503

# Command to run the Streamlit app on port 8503
CMD ["streamlit", "run", "app.py", "--server.port=8503"]