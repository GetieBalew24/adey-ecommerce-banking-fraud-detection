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