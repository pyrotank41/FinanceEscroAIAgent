# Use an official Python runtime as a parent image
FROM python:3.11.5

# Install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev curl && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the local code to the container
COPY . /app

# poetry:
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'

# Adjust the PATH environment variable
ENV PATH="/usr/local:$PATH"

CMD ls

# Install dependencies using Poetry
RUN poetry install 

# Check where Streamlit is installed
RUN find / -name streamlit

# Re-check the PATH and search for Streamlit
RUN echo $PATH && which streamlit

# Make port 8501 available to the world outside this container
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app when the container launches
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
