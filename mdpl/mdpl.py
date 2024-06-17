import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MedicalDataPipeLine(object):
    def __init__(self):
        self.df = pd.DataFrame()

    def load_medical_data(self, file_path_in):
        """
        This method converts a csv file to a pandas DataFrame

        Parameters:
        file_path_in : Source csv file paths (Ex : 'files/filename.csv')
        """
        try:
            dt_f = pd.read_csv(file_path_in)
            self.df = dt_f
            return self.df
        except Exception as e:
            print(f'please enter right file path! -> {e}')

    def save_medical_data(self, file_path_out):
        """
        This method converts a pandas DataFrame to a csv file and saves it to the specified file path

        Parameters:
        file_path_out : Result csv file paths (Ex : 'results/filename.csv')
        """
        try:
            self.df.to_csv(file_path_out, index=False)
        except Exception as e:
            print(f'please enter right file path! -> {e}') 

    def fill_nan_values(self, strategy='mean'):
        """
        This method fills the missing values in the DataFrame using the mean, median, and mode

        Parameters:
        strategy (str): Methods used to replace missing values (mean, median, mode)
        """
        try:
            for column in self.df.columns:
                if self.df[column].dtype != 'object':  # Check for non-string columns
                    if self.df[column].isnull().sum() > 0:
                        if strategy == 'mean':
                            self.df[column] = self.df[column].apply(lambda x : self.df[column].mean() if pd.isna(x) else x)
                        elif strategy == 'median':
                            self.df[column] = self.df[column].apply(lambda x : self.df[column].median() if pd.isna(x) else x)
                        elif strategy == 'mode':
                            self.df[column] = self.df[column].apply(lambda x : self.df[column].mode()[0] if pd.isna(x) else x)
        except Exception as e:
            print(f'You need to input a non-string data type! -> {e}')
        return self.df

    def normalize_data(self):
        """
        This method is used to standardize the data
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
        # Basic statistical calculations
        mean_column = self.df[column].mean()
        median_column = self.df[column].median()
        variance_column = self.df[column].var()
        std_dev_column = self.df[column].std()
        print(f"Mean Column: {mean_column}, Median Column: {median_column}, Variance Column: {variance_column}, Std Dev Column: {std_dev_column}")

    def distribution_analysis(self, column):
        """
        Distribution Analysis (Visualizing distributions using histograms and box plots)

        Parameters:
        column (str): Column name to which you want to apply distribution analysis
        """
        # Histogram
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df[column], kde=True, bins=10, color='blue')

        plt.title(f'{column} Distribution')
        plt.tight_layout()
        plt.show()
        
        # Box Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
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
        print('Correlation Matrix : {}'.format(correlation_matrix))
        
        # Heatmap Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    def distribution_visualization(self, column):
        """
        Distribution Visualization (Visualize distributions using kernel density estimation (KDE) plots)

        Parameters:
        column (str): Column name to which you want to apply distribution visualization
        """
        # Kernel density estimation (KDE) plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.kdeplot(self.df[column], fill=True, color='blue')
        plt.title(f'{column} KDE Plot')
        plt.tight_layout()
        plt.show()