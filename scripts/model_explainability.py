import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class FraudDetectionInterpretability:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.shap_explainer = None

    def load_and_split_data(self, test_size=0.2):
        """Load the dataset, split into features and target, and divide into training and testing sets."""
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['class'])  # Features
        y = data['class']  # Target variable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    def train_model(self):
        """Train the Random Forest model on the training data."""
        self.model.fit(self.X_train, self.y_train)