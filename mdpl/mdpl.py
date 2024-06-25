import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalDataPipeline(object):
    def __init__(self):
        self.df = pd.DataFrame()

    def load_medical_data(self, file_path_in):
        """
        This method converts a csv file to a pandas DataFrame

        Parameters:
        file_path_in : Source csv file path (Ex: 'files/filename.csv')
        """
        self.df = pd.read_csv(file_path_in)
        return self.df

    def save_medical_data(self, file_path_out):
        """
        This method converts a pandas DataFrame to a csv file and saves it to the specified file path

        Parameters:
        file_path_out : Result csv file path (Ex: 'results/filename.csv')
        """
        self.df.to_csv(file_path_out, index=False)

    def fill_nan_values(self, strategy='mean'):
        """
        This method fills the missing values in the DataFrame using the mean, median, or mode

        Parameters:
        strategy (str): Method used to replace missing values ('mean', 'median', 'mode')
        """
        if strategy not in ['mean', 'median', 'mode']:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        for column in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[column].isnull().sum() > 0:
                if strategy == 'mean':
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        return self.df

    def normalize_data(self):
        """
        This method standardizes the numeric data
        """
        scaler = StandardScaler()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns  # Only select numeric columns
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df

    def descriptive_statistics(self, column):
        """
        Descriptive Statistics (Basic statistical calculations - mean, median, variance, standard deviation)

        Parameters:
        column (str): Column name to which you want to apply descriptive statistics
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist in the DataFrame")

        # Basic statistical calculations
        stats = {
            "mean": self.df[column].mean(),
            "median": self.df[column].median(),
            "variance": self.df[column].var(),
            "std_dev": self.df[column].std()
        }
        return stats

    def distribution_analysis(self, column):
        """
        Distribution Analysis (Visualizing distributions using histograms and box plots)

        Parameters:
        column (str): Column name to which you want to apply distribution analysis
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist in the DataFrame")
        
        # Histogram
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df[column], kde=True, bins=10, color='blue')
        plt.title(f'{column} Distribution')
        
        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=self.df[column], color='blue')
        plt.title(f'{column} Box Plot')
        
        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """
        Correlation Analysis (Correlation analysis using correlation matrix and heatmap)
        """
        # Correlation Matrix Calculation
        correlation_matrix = self.df.corr()
        print('Correlation Matrix : \n{}'.format(correlation_matrix))
        
        # Heatmap Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()

    def distribution_visualization(self, column):
        """
        Distribution Visualization (Visualize distributions using kernel density estimation (KDE) plots)

        Parameters:
        column (str): Column name to which you want to apply distribution visualization
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist in the DataFrame")
        
        # Kernel density estimation (KDE) plot
        plt.figure(figsize=(12, 6))
        sns.kdeplot(self.df[column], fill=True, color='blue')
        plt.title(f'{column} KDE Plot')
        plt.tight_layout()
        plt.show()
