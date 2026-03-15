FROM python:3.11-slim

WORKDIR /app

COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt

COPY . .

RUN python data/generate_synthetic_transactions.py --n 50000 --out data/transactions_sample.csv && \
    python src/model_train.py --input data/transactions_sample.csv --eval --no-cache

EXPOSE 5000

CMD ["python", "src/dashboard.py"]