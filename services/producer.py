"""Simple Kafka producer that reads CSV and publishes one message per tx (JSON)."""
import argparse
import csv
import json
import time
from kafka import KafkaProducer


def produce(file, topic, bootstrap='localhost:9092', rate=0.0):
    producer = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            producer.send(topic, row)
            if rate>0:
                time.sleep(rate)
    producer.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--topic', default='transactions')
    parser.add_argument('--bootstrap', default='localhost:9092')
    parser.add_argument('--rate', type=float, default=0.0)
    args = parser.parse_args()
    produce(args.file, args.topic, args.bootstrap, args.rate)
