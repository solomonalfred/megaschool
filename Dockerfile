FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

CMD ["./start.sh"]version: "3.8"

services:
  fastapi-service:
    build: .
    container_name: fastapi-baseline
    ports:
      - "8080:8080"
    restart: unless-stopped
    environment:
      - TZ=UTC
    volumes:
      - ./logs:/app/logs
    # Если нужно GPU
    # runtime: nvidia
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]