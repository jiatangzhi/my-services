"""
predictor.py: this class is responsible for loading the artifacts and making the prediction.
"""

import os
import joblib
import torch
import pandas as pd
import logging as log
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.model import CreditScoringModel
from server.schemas import CreditRiskInput

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CreditRiskPredictor:
    """
    Orchestrates the loading of artifacts and the execution of inference.
    """
    def __init__(self, model_path: Path, preprocessor_path: Path, model_config: Dict[str, Any]):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_config = model_config
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """
        Load the artifacts from disk.
        """
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            log.info(f"✔ Preprocessor file loaded from: {self.preprocessor_path}")
        except FileNotFoundError:
            log.error(f"✘ Preprocessor file not found at {self.preprocessor_path}")
            raise
        
        try:
            # Recreate the model architecture
            self.model = CreditScoringModel(
                num_features=self.model_config['num_features'],  # This value must match the post-preprocessing feature size
                hidden_layers=self.model_config['hidden_layers'],
                dropout_rate=self.model_config['dropout_rate'],
                use_batch_norm=self.model_config['use_batch_norm'],
                activation_fn=self.model_config['activation_fn']
            )
            # Load trained weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()  # Set model to evaluation mode
            log.info(f"✔ Model weights loaded from: {self.model_path}")
            log.info("✔ Model and preprocessor successfully loaded.")
        except FileNotFoundError:
            log.error(f"✘ Model file not found at {self.model_path}")
            raise
        except Exception as e:
            log.error(f"✘ Error while loading the model: {e}")
            raise
        
    def predict(self, input_data: CreditRiskInput) -> Dict[str, Any]:
        """
        Perform a prediction.
        """
        # 1. Convert Pydantic input to DataFrame
        input_df = pd.DataFrame([input_data.dict(by_alias=True)])
        
        # 2. Apply preprocessing
        processed_features = self.preprocessor.transform(input_df)
        
        # 3. Convert to PyTorch tensor
        input_tensor = torch.tensor(processed_features, dtype=torch.float32)
        
        # 4. Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
        # 5. Format output
        prediction = 'good' if probability >= 0.5 else 'bad'
        log.info(f"✔ Prediction generated: {prediction} with probability: {probability:.4f}")
        
        return {
            "prediction": prediction,
            "probability": probability
        }
        

BEST_MODEL_CONFIG = {
    'num_features': 26, 
    'hidden_layers': [256, 128, 64, 64],
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    'activation_fn': 'ReLU'
}

# Paths relative to the project root `python/credit_scoring`
MODEL_PATH = Path("python/credit_scoring/models/mlp_credit_scoring_model_v1.3.0.pt") 
PREPROCESSOR_PATH = Path("python/credit_scoring/models/german_credit_risk_preprocessor.joblib")

# Singleton predictor instance to be used by the API.
# This ensures the model is loaded only once when the server starts.
predictor_instance = CreditRiskPredictor(
    model_path=MODEL_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    model_config=BEST_MODEL_CONFIG
)
