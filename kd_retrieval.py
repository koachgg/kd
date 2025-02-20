import os
import yaml
import logging
from datetime import datetime
import torch
from utils.data_loader import SciFactDataLoader
from utils.model_trainer import KDTrainer
from utils.evaluator import RetrievalEvaluator

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging_file = os.path.join('logs', f'kd_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_file),
            logging.StreamHandler()
        ]
    )

def load_config():
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise

def main():
    try:
        # Setup
        setup_logging()
        logging.info(f"Starting KD process at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"User: koachgg")
        
        # Load config and initialize components
        config = load_config()
        os.makedirs(config['output']['model_dir'], exist_ok=True)
        
        data_loader = SciFactDataLoader(config)
        trainer = KDTrainer(config)
        evaluator = RetrievalEvaluator(config)
        
        # Load data
        logging.info("Loading data...")
        corpus, queries, qrels = data_loader.load_data()
        
        # Train model
        logging.info("Starting training...")
        student_model = trainer.distill(
            teacher_model=config["models"]["teacher"]["name"],
            student_model=config["models"]["student"]["name"],
            queries=queries,
            qrels_train=qrels["train"],
            corpus=corpus
        )
        
        # Evaluate
        logging.info("Evaluating model...")
        metrics = evaluator.evaluate(
            model=student_model,
            test_data=qrels["test"],
            queries=queries,
            corpus=corpus
        )
        
        # Save results
        try:
            import pandas as pd
            results = {
                'Experiment ID': 1,
                'Teacher Model': config["models"]["teacher"]["name"],
                'Student Model': config["models"]["student"]["name"],
                'Dataset': config["dataset"]["name"],
                'DCG': metrics['DCG'],
                'NDCG': metrics['NDCG'],
                'Recall@10': metrics['Recall@10'],
                'MAP@10': metrics['MAP@10'],
                'Training Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Hardware': 'GPU' if torch.cuda.is_available() else 'CPU'
            }
            
            results_path = config['output']['results_file']
            if os.path.exists(results_path):
                df = pd.read_excel(results_path)
                results['Experiment ID'] = len(df) + 1
                df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
            else:
                df = pd.DataFrame([results])
            
            df.to_excel(results_path, index=False)
            logging.info(f"Results saved to {results_path}")
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()