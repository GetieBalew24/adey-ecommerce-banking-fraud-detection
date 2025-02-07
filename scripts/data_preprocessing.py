import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logg = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path1, file_path2, file_path3):
        """
        Initialize the DataPreprocessor with file paths for three datasets.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.data = None
        self.data1 = None
        self.data2 = None

    def load_data(self):
        """
        Load data from the provided file paths into Pandas DataFrames.
        """
        try:
            self.data = pd.read_csv(self.file_path1)
            self.data1 = pd.read_csv(self.file_path2)
            self.data2 = pd.read_csv(self.file_path3)
            logg.info("Data loaded successfully!")
            return self.data, self.data1, self.data2
        
        except Exception as e:
            logg.error(f"An error occurred while loading data: {e}")
            return None, None, None
    def explore_data(self):
        """
        Display the first few rows of the main dataset for exploration purposes.
        """
        logg.info("Exploring data:")
        logg.info(f"\n{self.data.head()}")

    def check_missing_values(self, data):
        """
        Check for missing values in the provided dataset.
        """
        missing_values = data.isnull().sum()
        logg.info(f"Missing values: \n{missing_values}")
        return missing_values
    def handle_missing_values(self):
        """
        Handle missing values in numerical columns by imputing with the column mean.
        """
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.data[num_cols].isnull().sum().any():
            self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].mean())
            logg.info("Missing values in numerical columns handled with mean imputation.")
        else:
            logg.info("No missing values found in numerical columns.")
        return self.data
    def feature_engineering(self, merged_data):
        """
        Perform feature engineering on the dataset by creating new features based on existing columns.
        """
        logg.info("Performing feature engineering...")

        # Ensure 'purchase_time' is a valid datetime
        merged_data['purchase_time'] = pd.to_datetime(merged_data['purchase_time'], errors='coerce')

        # Create 'hour_of_day' and 'day_of_week' features
        merged_data['hour_of_day'] = merged_data['purchase_time'].dt.hour
        merged_data['day_of_week'] = merged_data['purchase_time'].dt.dayofweek  # 0=Monday, 6=Sunday

        logg.info("Feature engineering completed: added 'hour_of_day' and 'day_of_week'.")
        
        # Calculate transaction frequency and velocity
        # self.calculate_transaction_features(merged_data)
        
        return merged_data
