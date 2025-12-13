<<<<<<< HEAD
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "TRADERSUN.py"]
=======
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "TRADERSUN.py"]
>>>>>>> 0c9476172d66818fb9c746e888d32dbb0a82c0f3
