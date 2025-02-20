import json
import pandas as pd
from qdrant_client import QdrantClient
from tqdm import tqdm
import logging
from datetime import datetime

class SciFactDataLoader:
    def __init__(self, config):
        self.config = config
        self.qdrant_client = QdrantClient(
            host=config["qdrant"]["host"],
            port=config["qdrant"]["port"]
        )
        self.setup_logger()

    def setup_logger(self):
        logging.info(f"Data loader initialized at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"User: koachgg")

    def load_jsonl(self, file_path):
        """Load JSONL file with _id field"""
        data = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Use _id instead of id
                    data[str(item['_id'])] = item['text']  # Convert _id to string for consistency
            return data
        except Exception as e:
            logging.error(f"Error loading JSONL file {file_path}: {str(e)}")
            logging.error(f"First line of file: {next(open(file_path))}")  # Debug info
            raise

    def load_qrels(self):
        """Load train and test TSV files"""
        try:
            train_qrels = pd.read_csv(
                f"{self.config['dataset']['paths']['qrels']}/train.tsv",
                sep='\t',
                names=['query-id', 'corpus-id', 'score']
            )
            
            test_qrels = pd.read_csv(
                f"{self.config['dataset']['paths']['qrels']}/test.tsv",
                sep='\t',
                names=['query-id', 'corpus-id', 'score']
            )
            
            # Convert IDs to strings for consistency
            train_qrels['query-id'] = train_qrels['query-id'].astype(str)
            train_qrels['corpus-id'] = train_qrels['corpus-id'].astype(str)
            test_qrels['query-id'] = test_qrels['query-id'].astype(str)
            test_qrels['corpus-id'] = test_qrels['corpus-id'].astype(str)
            
            return {
                "train": train_qrels,
                "test": test_qrels
            }
        except Exception as e:
            logging.error(f"Error loading qrels: {str(e)}")
            raise

    def validate_data(self, corpus, queries, qrels):
        """Validate data consistency"""
        try:
            # Check if all query IDs in qrels exist in queries
            train_query_ids = set(qrels['train']['query-id'])
            test_query_ids = set(qrels['test']['query-id'])
            query_ids = set(queries.keys())
            
            missing_train_queries = train_query_ids - query_ids
            missing_test_queries = test_query_ids - query_ids
            
            if missing_train_queries:
                logging.warning(f"Missing queries in training set: {missing_train_queries}")
            if missing_test_queries:
                logging.warning(f"Missing queries in test set: {missing_test_queries}")
            
            # Check if all corpus IDs in qrels exist in corpus
            train_corpus_ids = set(qrels['train']['corpus-id'])
            test_corpus_ids = set(qrels['test']['corpus-id'])
            corpus_ids = set(corpus.keys())
            
            missing_train_docs = train_corpus_ids - corpus_ids
            missing_test_docs = test_corpus_ids - corpus_ids
            
            if missing_train_docs:
                logging.warning(f"Missing documents in training set: {missing_train_docs}")
            if missing_test_docs:
                logging.warning(f"Missing documents in test set: {missing_test_docs}")
            
        except Exception as e:
            logging.error(f"Error validating data: {str(e)}")
            raise

    def load_data(self):
        """Load all required data"""
        try:
            # Load queries (JSONL format)
            logging.info("Loading queries...")
            queries = self.load_jsonl(self.config["dataset"]["paths"]["queries"])
            logging.info(f"Loaded {len(queries)} queries")
            
            # Load corpus (JSONL format)
            logging.info("Loading corpus...")
            corpus = self.load_jsonl(self.config["dataset"]["paths"]["corpus"])
            logging.info(f"Loaded {len(corpus)} documents")
            
            # Load qrels
            logging.info("Loading qrels...")
            qrels = self.load_qrels()
            logging.info(f"Loaded {len(qrels['train'])} training pairs and {len(qrels['test'])} test pairs")
            
            # Validate data
            self.validate_data(corpus, queries, qrels)
            
            return corpus, queries, qrels
            
        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise

    def print_sample_data(self, corpus, queries, qrels):
        """Print sample data for debugging"""
        try:
            logging.info("\nSample data:")
            logging.info("\nFirst query:")
            first_query_id = next(iter(queries))
            logging.info(f"ID: {first_query_id}")
            logging.info(f"Text: {queries[first_query_id]}")
            
            logging.info("\nFirst corpus document:")
            first_doc_id = next(iter(corpus))
            logging.info(f"ID: {first_doc_id}")
            logging.info(f"Text: {corpus[first_doc_id]}")
            
            logging.info("\nFirst training pair:")
            logging.info(qrels['train'].iloc[0].to_dict())
            
        except Exception as e:
            logging.error(f"Error printing sample data: {str(e)}")
