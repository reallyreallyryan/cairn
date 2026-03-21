FROM python:3.11-slim

# Pre-install common libraries the agent might need
RUN pip install --no-cache-dir \
    requests httpx beautifulsoup4 lxml \
    pyyaml toml \
    markdown

# Create sandbox directories
RUN mkdir -p /sandbox/workspace /sandbox/output && \
    chmod 777 /sandbox/workspace /sandbox/output /tmp

# Non-root user for execution
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /sandbox/workspace
