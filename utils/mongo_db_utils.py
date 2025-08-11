from pymongo import MongoClient
import logging

class MongoHandler(logging.Handler):
    def __init__(self, db_name, collection_name, uri="mongodb://localhost:27017/"):
        super().__init__()
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def emit(self, record):
        log_entry = self.format(record)
        self.collection.insert_one({"log": log_entry, "level": record.levelname})
