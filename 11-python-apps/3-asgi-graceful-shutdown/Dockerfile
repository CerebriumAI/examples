# Base image
FROM python:3.12-bookworm
RUN apt-get update
RUN update-ca-certificates

# Source code
COPY . .

# dependencies
RUN pip install -r requirements.txt

# Configuration - make sure this matches your cerebrium.toml port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
