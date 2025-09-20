FROM openjdk:17-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
