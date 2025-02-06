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
    def plot_bar_chart(self, categorical_features: list):
        """Plots bar charts for each specified categorical feature in a grid layout.

        Args:
            categorical_features (list): List of categorical feature names to plot.
        """
        try:
            num_features = len(categorical_features)
            num_cols = 2  # We want 2 columns
            num_rows = (num_features + num_cols - 1) // num_cols
            
            plt.figure(figsize=(num_cols * 6, num_rows * 4))

            for i, feature in enumerate(categorical_features, 1):
                if feature not in self.data.columns:
                    logg.error(f"Feature '{feature}' not found in data!")
                    continue

                plt.subplot(num_rows, num_cols, i)
                sns.barplot(
                    x=self.data[feature].value_counts().index,
                    y=self.data[feature].value_counts().values,
                    palette='viridis'
                )
                plt.title(f'Bar Chart for {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

            logg.info("Bar charts plotted successfully!")
        except Exception as e:
            logg.error(f"An error occurred while plotting bar charts: {e}")