# Use a Python version consistent with the project (e.g., 3.11)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --system kedro && \
    useradd --system --gid kedro --shell /bin/bash --home /home/kedro kedro

RUN mkdir -p /app/data /app/logs && \
    chown -R kedro:kedro /app/data /app/logs

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY conf conf

RUN chown -R kedro:kedro /app

USER kedro
WORKDIR /app

ENTRYPOINT ["kedro"]

CMD ["run"]
