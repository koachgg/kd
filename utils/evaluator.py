import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from tqdm import tqdm
import logging
from datetime import datetime
import torch

class RetrievalEvaluator:
    def __init__(self, config):
        try:
            self.qdrant = QdrantClient(
                host=config["qdrant"]["host"],
                port=config["qdrant"]["port"]
            )
            self.config = config
            self.collection_name = config["qdrant"]["collection"]
            self.setup_logger()
            
        except Exception as e:
            logging.error(f"Error initializing evaluator: {str(e)}")
            raise

    
    def setup_logger(self):
        # Log initialization time in UTC
        logging.info(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Current User's Login: Belo")
        logging.info(f"Evaluator initialized at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Connected to Qdrant at {self.config['qdrant']['host']}:{self.config['qdrant']['port']}")

    def upload_corpus_embeddings(self, model, corpus):
        """Upload corpus embeddings to Qdrant"""
        try:
            logging.info("Starting corpus embedding upload...")
            
            # Create or recreate collection
            collections = self.qdrant.get_collections().collections
            if any(col.name == self.collection_name for col in collections):
                self.qdrant.delete_collection(self.collection_name)
            
            # Get corpus embeddings
            corpus_texts = list(corpus.values())
            corpus_ids = list(corpus.keys())
            
            # Create ID mapping (original ID -> integer ID)
            self.id_mapping = {original_id: idx for idx, original_id in enumerate(corpus_ids)}
            
            # Get vector size from a sample encoding
            sample_embedding = model.encode([corpus_texts[0]], convert_to_tensor=True)
            vector_size = sample_embedding.shape[1]
            
            # Create collection with proper vector size
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            
            # Process in batches
            batch_size = 32
            for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Uploading corpus"):
                batch_texts = corpus_texts[i:i + batch_size]
                batch_original_ids = corpus_ids[i:i + batch_size]
                
                # Encode batch
                batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
                batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Create points with integer IDs
                points = [
                    PointStruct(
                        id=self.id_mapping[original_id],  # Use integer ID
                        vector=embedding.tolist(),
                        payload={
                            "text": text,
                            "original_id": str(original_id)  # Store original ID in payload
                        }
                    )
                    for original_id, embedding, text in zip(batch_original_ids, batch_embeddings, batch_texts)
                ]
                
                # Upload batch
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            # Verify upload
            collection_info = self.qdrant.get_collection(self.collection_name)
            logging.info(f"Uploaded {collection_info.points_count} vectors to Qdrant")
            
        except Exception as e:
            logging.error(f"Error uploading corpus embeddings: {str(e)}")
            raise

    def calculate_dcg(self, relevances, k=10):
        """Calculate DCG@k"""
        dcg = 0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def calculate_ndcg(self, retrieved_docs, relevant_docs, k=10):
        """Calculate NDCG@k"""
        relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        dcg = self.calculate_dcg(relevances, k)
        ideal_relevances = sorted([1 if doc in relevant_docs else 0 for doc in retrieved_docs], reverse=True)
        idcg = self.calculate_dcg(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0

    def calculate_map(self, retrieved_docs, relevant_docs, k=10):
        """Calculate MAP@k"""
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_docs:
                relevant_count += 1
                precisions.append(relevant_count / i)
                
        return np.mean(precisions) if precisions else 0

    def evaluate(self, model, test_data, queries, corpus):
        """Evaluate the model using test data"""
        try:
            logging.info("Starting evaluation...")
            
            # Upload corpus embeddings to Qdrant
            self.upload_corpus_embeddings(model, corpus)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            metrics = {
                'dcg': [],
                'ndcg': [],
                'recall@10': [],
                'map@10': []
            }

            # Evaluate each query
            for query_id, group in tqdm(test_data.groupby('query-id'), desc="Evaluating queries"):
                if query_id not in queries:
                    continue

                query_text = queries[query_id]
                relevant_docs = set(group['corpus-id'].astype(str).tolist())

                # Search in Qdrant
                try:
                    query_embedding = model.encode(query_text, convert_to_tensor=True, device=device)
                    search_results = self.qdrant.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding.cpu().numpy(),
                        limit=10  # Only get top 10 results since we're calculating @10 metrics
                    )
                    
                    
                    retrieved_docs = [hit.payload["original_id"] for hit in search_results]
                    
                except Exception as e:
                    logging.error(f"Error searching Qdrant for query {query_id}: {str(e)}")
                    continue

                # Calculate all metrics @10
                relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
                dcg = self.calculate_dcg(relevances)
                ndcg = self.calculate_ndcg(retrieved_docs, relevant_docs)
                recall = len(set(retrieved_docs) & relevant_docs) / len(relevant_docs) if relevant_docs else 0
                map_score = self.calculate_map(retrieved_docs, relevant_docs)

                metrics['dcg'].append(dcg)
                metrics['ndcg'].append(ndcg)
                metrics['recall@10'].append(recall)
                metrics['map@10'].append(map_score)

            # Calculate final metrics
            final_metrics = {
                'DCG': np.mean(metrics['dcg']),
                'NDCG': np.mean(metrics['ndcg']),
                'Recall@10': np.mean(metrics['recall@10']),
                'MAP@10': np.mean(metrics['map@10'])
            }

            logging.info("\nEvaluation Results:")
            for metric, value in final_metrics.items():
                logging.info(f"{metric}: {value:.3f}")

            return final_metrics

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def _calculate_dcg(self, relevances, k=10):
        """Calculate DCG@k"""
        dcg = 0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def _calculate_ndcg(self, retrieved_docs, relevant_docs, k=10):
        """Calculate NDCG@k"""
        relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        dcg = self._calculate_dcg(relevances, k)
        ideal_relevances = sorted([1 if doc in relevant_docs else 0 for doc in retrieved_docs], reverse=True)
        idcg = self._calculate_dcg(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0

    def _calculate_map(self, retrieved_docs, relevant_docs, k=10):
        """Calculate MAP@k"""
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs[:k], 1):
            if doc in relevant_docs:
                relevant_count += 1
                precisions.append(relevant_count / i)
                
        return np.mean(precisions) if precisions else 0