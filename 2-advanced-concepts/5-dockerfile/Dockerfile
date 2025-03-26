# Base image
FROM python:3.12-bookworm
RUN apt-get update && apt-get install dumb-init
RUN update-ca-certificates

# Source code
COPY . .

# Dependencies
RUN pip install -r requirements.txt

# Configuration
EXPOSE 8192
CMD ["dumb-init", "--", "fastapi", "run", "main.py", "--port", "8192"]