import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging as logg

logg.basicConfig(level=logg.INFO)

class DataVisualizer:
    """
    A class for visualizing data using various plotting techniques.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing the dataset to be visualized.
    """

    def __init__(self, data):
        """
        Initializes the DataVisualizer with the provided dataset.

        Args:
            data (pd.DataFrame): The dataset to visualize.
        """
        self.data = data

    def visualize_data(self):
        """Visualizes the data using a pairplot to show relationships between features."""
        sns.pairplot(self.data)
        plt.show()
    def plot_histogram(self, numerical_features):
        """Plots histograms for the specified numerical features.

        Args:
            numerical_features (list): List of numerical feature names to plot.
        """
        plt.figure(figsize=(16, 5))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(1, len(numerical_features), i)
            sns.histplot(self.data[feature], bins=20, kde=True)
            plt.title(f'Histogram for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        logg.info("Histograms plotted successfully!")