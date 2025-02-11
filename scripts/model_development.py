import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import logging

# Create log directory if not exists
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging to log both to a file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/pipeline.log"),
        logging.StreamHandler()
    ]
)

# Set MLflow tracking URI to the root directory
mlflow.set_tracking_uri("/home/gech/10 acadamy/week 8-9/project/adey-ecommerce-banking-fraud-detection/mlruns")


class ModelPipeline:
    """Class to handle data loading, splitting, model training, evaluation, and logging."""

    def __init__(self, dataset_type, path):
        """
        Initialize the pipeline with dataset type and file path.
        
        dataset_type: A string ('creditcard' or 'fraud') to indicate dataset type.
        path: File path for the dataset.
        """
        self.dataset_type = dataset_type
        self.path = path
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set different experiments based on the dataset
        if self.dataset_type == 'creditcard':
            self.experiment_name = 'creditcard_experiment'
        elif self.dataset_type == 'fraud':
            self.experiment_name = 'Fraud_Detection_Experiment'
        else:
            raise ValueError("Invalid dataset_type! Must be 'creditcard' or 'fraud'")

        # Set the experiment for MLflow
        mlflow.set_experiment(self.experiment_name)
    def load_data(self):
        """Load data based on the dataset type."""
        if self.dataset_type == 'creditcard':
            logging.info(f"Loading credit card data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'Class'  # Target column for creditcard dataset

        elif self.dataset_type == 'fraud':
            logging.info(f"Loading fraud data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'class'  # Target column for fraud dataset

        else:
            raise ValueError("Invalid dataset_type! Must be 'creditcard' or 'fraud'")
        
        logging.info("Data loading complete.")