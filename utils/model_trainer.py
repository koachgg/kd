import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from datetime import datetime
import os
import time

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class TrainingDataset(Dataset):
    def __init__(self, queries, corpus, qrels):
        self.data = []
        
        for _, row in qrels.iterrows():
            query_id = str(row['query-id'])
            doc_id = str(row['corpus-id'])
            if query_id in queries and doc_id in corpus:
                self.data.append({
                    'query_text': queries[query_id],
                    'doc_text': corpus[doc_id],
                    'score': float(row['score'])
                })
        
        logging.info(f"Created dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class KDTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["training"].get("device", "cpu"))
        self.setup_logger()
        self.initialize_training()

    def setup_logger(self):
        logging.info(f"KD Trainer initialized at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"User: koachgg")
        logging.info(f"Device: {self.device}")

    def initialize_training(self):
        try:
            self.teacher_model_name = self.config["models"]["teacher"]["name"]
            self.student_model_name = self.config["models"]["student"]["name"]
            
            # Convert learning rates to float
            self.learning_rates = {
                'student': float(self.config["training"]["learning_rate"]["student"]),
                'projection': float(self.config["training"]["learning_rate"]["projection"])
            }
            
            logging.info("Initialized training with:")
            logging.info(f"Teacher model: {self.teacher_model_name}")
            logging.info(f"Student model: {self.student_model_name}")
            logging.info(f"Learning rates: {self.learning_rates}")
        except Exception as e:
            logging.error(f"Error initializing training: {str(e)}")
            raise

    def distill(self, teacher_model, student_model, queries, qrels_train, corpus):  # Added corpus parameter
        try:
            start_time = time.time()
            logging.info(f"Starting distillation at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

            # Initialize models
            teacher = SentenceTransformer(teacher_model)
            student = SentenceTransformer(student_model)

            # Get dimensions
            teacher_dim = teacher.get_sentence_embedding_dimension()
            student_dim = student.get_sentence_embedding_dimension()
            logging.info(f"Teacher dimension: {teacher_dim}, Student dimension: {student_dim}")

            # Initialize projection head
            projection = ProjectionHead(student_dim, teacher_dim).to(self.device)

            # Initialize optimizers with float learning rates
            student_optimizer = torch.optim.AdamW(
                student.parameters(),
                lr=self.learning_rates['student']
            )
            projection_optimizer = torch.optim.AdamW(
                projection.parameters(),
                lr=self.learning_rates['projection']
            )

            # Create dataset using the passed corpus parameter
            train_dataset = TrainingDataset(queries, corpus, qrels_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True
            )

            # Training loop
            teacher.to(self.device)
            student.to(self.device)

            for epoch in range(self.config["training"]["epochs"]):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
                
                for batch in progress_bar:
                    # Get embeddings
                    with torch.no_grad():
                        teacher_embeds = teacher.encode(
                            batch['query_text'],
                            convert_to_tensor=True,
                            device=self.device
                        )

                    student_embeds = student.encode(
                        batch['query_text'],
                        convert_to_tensor=True,
                        device=self.device
                    )

                    # Project student embeddings
                    projected_student = projection(student_embeds)

                    # Calculate loss
                    loss = nn.MSELoss()(projected_student, teacher_embeds)

                    # Backward pass
                    student_optimizer.zero_grad()
                    projection_optimizer.zero_grad()
                    loss.backward()
                    student_optimizer.step()
                    projection_optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                avg_loss = total_loss / len(train_loader)
                logging.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            # Save models
            output_dir = self.config["output"]["model_dir"]
            os.makedirs(output_dir, exist_ok=True)
            student.save(os.path.join(output_dir, 'student_model'))
            torch.save(projection.state_dict(), os.path.join(output_dir, 'projection_head.pt'))

            training_time = time.time() - start_time
            logging.info(f"Training completed in {training_time:.2f} seconds")

            return student

        except Exception as e:
            logging.error(f"Error in distillation process: {str(e)}")
            raise