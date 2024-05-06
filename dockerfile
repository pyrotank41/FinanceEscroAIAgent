# Use an official Python runtime as a parent image
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies required for Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the Poetry configuration files into the container
COPY pyproject.toml poetry.lock* ./

# Configure Poetry:
# - Disable virtual environments as the container itself is isolated
# - Install all dependencies including dev if needed
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY prompts ./prompts
COPY utility ./utility
COPY escrow_data ./escrow_data
COPY chat_assistant.py app.py rag.py .env ./

# The command to run the application
# CMD ["poetry", "run", "python", "chat_assistant.py"]
