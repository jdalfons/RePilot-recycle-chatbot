# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y gcc 
RUN pip install --no-cache-dir -r requirements.txt

# Ensure /tmp is writable
RUN chmod -R 777 /tmp

# Copy the rest of the application code into the container
COPY . .

ENV MONGO_HOST=${MONGO_HOST:-localhost}
ENV POSTGRES_HOST=${POSTGRES_HOST:-localhost}
ENV POSTGRES_USER=${POSTGRES_USER:-llm}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-llm}
ENV POSTGRES_DBNAME=${POSTGRES_DBNAME:-llm}
ENV POSTGRES_PORT=${POSTGRES_PORT:-5432}
ENV CHROMA_SERVER=${CHROMA_SERVER:-localhost}
ENV CHROMA_PORT=${CHROMA_PORT:-32003}
ENV MISTRAL_API_KEY=${MISTRAL_API_KEY}
ENV HF_TOKEN=${HF_TOKEN}

# Expose the port that Streamlit will run on
EXPOSE 8503

# Command to run the Streamlit app on port 8503
CMD ["streamlit", "run", "app.py", "--server.port=8503"]