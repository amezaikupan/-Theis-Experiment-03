import pandas as pd
import numpy as np 
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data, task_division, target, categorical_features=None, numerical_features=None, fill_na=False):
        self.target = target
        self.task_division = task_division
        self.data = data
        self.fill_na = fill_na

        print(' **** BASIC DATA STATISTICS **** ')
        print(f'Number of columns: {len(data.columns)}')
        print(f'Number of rows: {len(data)}')

        print('\n **** PROCESS NA **** ')
        # print('Before processing data:')
        cols_removed = self.process_na_cols()
        self.numerical_features = [feat for feat in numerical_features if feat not in cols_removed]
        self.categorical_features = [col for col in self.data.columns
                if col not in numerical_features
                and col not in task_division
                and col != target]
        
        # print("Columns removed due to high NA percentage:")
        # print(cols_removed)
        self.process_na_rows()
        print("Complete processing NA!")

        # print('\n **** PROCESS CATEGORICAL & NUMERICAL COLUMNS **** ')
        # print(f"Number of columns before processing: {len(self.data.columns)}")
        # self.process_features()
        # print(f"Number of columns after processing: {len(self.data.columns)}")
        # # Remove outliners 

        # self.log_scale_features(self.data)

        print("\n*** Divide tasks ***")
        # self.clip_extreme_values(self.target)
        self.task_division = task_division
        self.tasks = self._task_division()
        self.n_tasks = len(self.tasks)

        self.data = self.data.drop(columns=task_division)
        print(f"Complete dropping task division: {task_division}")
    
    def __del__(self):
        """
        Destructor method called when object is about to be garbage collected.
        Performs explicit cleanup of large data structures.
        """
        try:
            self.cleanup()
            print("DataProcessor object cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def cleanup(self):
        """
        Explicit cleanup method to free memory and resources.
        Call this when you're done with the object.
        """
        # Clear large data structures
        if hasattr(self, 'data'):
            del self.data
        
        if hasattr(self, 'tasks'):
            for task_name in list(self.tasks.keys()):
                del self.tasks[task_name]
            del self.tasks
        
        if hasattr(self, 'train_tasks'):
            for task_name in list(self.train_tasks.keys()):
                del self.train_tasks[task_name]
            del self.train_tasks
        
        if hasattr(self, 'test_tasks'):
            for task_name in list(self.test_tasks.keys()):
                del self.test_tasks[task_name]
            del self.test_tasks
        
        # Clear other attributes
        self.categorical_features = None
        self.numerical_features = None
        self.target = None
        self.task_division = None
        
        # Force garbage collection
        gc.collect()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup automatically"""
        self.cleanup()

    def get_memory_usage(self):
        """
        Get approximate memory usage of the object's data structures.
        """
        total_memory = 0
        
        if hasattr(self, 'data') and self.data is not None:
            total_memory += self.data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'tasks') and self.tasks:
            for task_data in self.tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'train_tasks') and self.train_tasks:
            for task_data in self.train_tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'test_tasks') and self.test_tasks:
            for task_data in self.test_tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        return total_memory / (1024**2)  # Return in MB
    
    def _task_division(self):
        # print('\n **** TASK DIVISION **** ')

        # Implement task division logic here
        tasks = {}
        groups = self.data.groupby(self.task_division)
        for name, group in groups:
            tasks[name] = group.copy().reset_index(drop=True).drop(columns=self.task_division)
        return tasks 
    
    def process_na_cols(self, remove_col_na_thres = 0.4):
        # Calculate NA percentage per column
        col_na_percent = self.data.isna().sum() / len(self.data)
        
        # Find columns to remove
        remove_cols = col_na_percent[col_na_percent > remove_col_na_thres].index.tolist()
        
        # Remove high-NA columns
        if remove_cols:
            self.data = self.data.drop(columns=remove_cols)
        
        return remove_cols
  
    def process_na_rows(self):

        self.data = self.data.dropna(subset=[self.target])

        if self.fill_na:
            # Fill NaN values in numerical columns with their mean
            self.data[self.numerical_features] = self.data[self.numerical_features].fillna(
                self.data[self.numerical_features].mean()
            )
            
            # Fill NaN values in categorical columns with the most frequent value
            self.data[self.categorical_features] = self.data[self.categorical_features].fillna(
                self.data[self.categorical_features].mode().iloc[0]
            )
        else:
            # Drop any row with missing values
            self.data = self.data.dropna(how="any")

    def z_score_outliers(self, data, threshold=3):
        """
        Detect outliers using Z-score method
        
        Parameters:
        threshold: Z-score threshold (default: 3)
        
        Returns:
        Dictionary with outlier indices for each column
        """
        outliers = {}
        
        for col in self.numerical_features:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers[col] = outlier_indices
            
        return outliers
    
    def iqr_outliers(self, data):
        outliers = {}
        for col in self.numerical_features:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            outlier_indices = data[(data[col] < lower_bound) | 
                                (data[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices
        return outliers

    def remove_outliers(self, data, method='zscore', columns=None):
        df_clean = data.copy()
        if columns is None:
            columns = df_clean.select_dtypes(include='number').columns

        outlier_indices = set()
        
        if method == 'iqr':
            outliers = self.iqr_outliers(df_clean)
        elif method == 'zscore':
            outliers = self.z_score_outliers(df_clean)
        else:
            raise ValueError(f"Unknown method '{method}'")
        
        for col in columns:
            if col in outliers:
                outlier_indices.update(outliers[col])
        
        valid_indices = df_clean.index.intersection(outlier_indices)
        df_clean = df_clean.drop(index=valid_indices)

        print(f"Original dataset size: {len(data)}")
        print(f"Cleaned dataset size: {len(df_clean)}")
        print(f"Removed {len(valid_indices)} outliers ({len(valid_indices)/len(data)*100:.2f}%)")
        
        return df_clean

    def log_scale_features(self, data):
        data = data.copy()
        for column in data.select_dtypes(include='number').columns:
            data[column] = np.log1p(data[column])
        return data

    def train_test_split(self, test_split=0.3, random_state=42):
        task_names = list(self.tasks.keys())
        task_names_sorted = task_names.copy()
        
        import random
        random.seed(random_state)
        random.shuffle(task_names_sorted)
        
        n_tasks = len(task_names_sorted)
        n_test_tasks = max(1, int(n_tasks * test_split))
        
        train_task_names = task_names_sorted[:-n_test_tasks]
        test_task_names = task_names_sorted[-n_test_tasks:]
        
        train_tasks = {name: self.tasks[name] for name in train_task_names}
        test_tasks = {name: self.tasks[name] for name in test_task_names}
        
        # Fit scaler
        X_train_all = np.vstack([
            task.drop(columns=[self.target]).values
            for task in train_tasks.values()
        ])
        scaler = StandardScaler()
        scaler.fit(X_train_all)
        
        # Process train tasks
        for name, task in train_tasks.items():
            # Remove outliers 
            task = self.remove_outliers(task)

            # Log transform and restore target
            log_data = self.log_scale_features(task.drop(columns=[self.target]))
            log_data[self.target] = task[self.target]
            train_tasks[name] = log_data
            
            # Standard scale
            X = task.drop(columns=[self.target])
            y = task[self.target]
            X_scaled = scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=task.index)
            task_scaled = X_scaled_df.copy()
            task_scaled[self.target] = y

        # Process test tasks
        for name, task in test_tasks.items():
            # Log transform and restore target 
            log_data = self.log_scale_features(task_scaled.drop(columns=[self.target]))
            log_data[self.target] = task_scaled[self.target]
            test_tasks[name] = log_data

            # Standard scale
            X = task.drop(columns=[self.target])
            y = task[self.target]
            X_scaled = scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=task.index)
            task_scaled = X_scaled_df.copy()
            task_scaled[self.target] = y

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        return train_tasks, test_tasks
    
    def top_k_train_cv(self, k, random_order=False, random_state=42):
        """
        Generator for Cross-Validation where the model is trained on the top-k tasks 
        (with the most data points) and tested on each of the remaining tasks individually.
        
        Parameters:
        - k: int, number of top tasks (with most data) to use for training
        - random_order: bool, whether to randomize the test task order
        - random_state: int, random seed for reproducibility
        
        Yields:
        (train_tasks, test_task, train_task_names, test_task_name): tuple
        """
        # Compute number of examples for each task
        # Assuming tasks are stored as dictionaries with data directly accessible
        task_lengths = {name: len(data) for name, data in self.tasks.items()}
        
        # Sort tasks by number of examples (descending)
        sorted_tasks = sorted(task_lengths.items(), key=lambda item: item[1], reverse=True)
        
        # Select top-k tasks for training
        train_task_names = [name for name, _ in sorted_tasks[:k]]
        test_task_names = [name for name, _ in sorted_tasks[k:]]

        if random_order:
            import random
            rng = random.Random(random_state)
            rng.shuffle(test_task_names)

        # Create training set once
        train_tasks = {name: self.tasks[name] for name in train_task_names}

        for test_task_name in test_task_names:
            test_task = {test_task_name: self.tasks[test_task_name]}
            yield train_tasks, test_task, train_task_names, test_task_name

    def get_top_k_split(self, k, random_order=False, random_state=42):
        """
        Split tasks into top-k tasks for training and remaining tasks for testing.
        
        Parameters:
        - k: int, number of top tasks (with most data) to use for training
        - random_order: bool, whether to randomize the test task order
        - random_state: int, random seed for reproducibility
        
        Returns:
        (train_tasks, test_tasks_list, train_task_names, test_task_names): tuple
        - train_tasks: dict, top-k tasks for training
        - test_tasks_list: list of dict, each dict contains one test task
        - train_task_names: list of str, names of training tasks
        - test_task_names: list of str, names of test tasks
        """
        # Compute number of examples for each task
        # Assuming tasks are stored as dictionaries with data directly accessible
        task_lengths = {name: len(data) for name, data in self.tasks.items()}
        
        # Sort tasks by number of examples (descending)
        sorted_tasks = sorted(task_lengths.items(), key=lambda item: item[1], reverse=True)
        
        # Select top-k tasks for training
        train_task_names = [name for name, _ in sorted_tasks[:k]]
        test_task_names = [name for name, _ in sorted_tasks[k:]]

        if random_order:
            import random
            rng = random.Random(random_state)
            rng.shuffle(test_task_names)

        # Create training set (all top-k tasks combined)
        train_tasks = {name: self.tasks[name] for name in train_task_names}
        
        # Create test tasks list (each task as separate dict for individual testing)
        test_tasks_list = [{name: self.tasks[name]} for name in test_task_names]

        return train_tasks, test_tasks_list, train_task_names, test_task_names

    def leave_percent_tasks_out_cv(self, percent=0.1, random_order=False, random_state=42):
        """
        Generator for Leave-Percent-of-Tasks-Out Cross Validation (LPT-CV).
        In each iteration, a percentage of tasks are held out as the test set, 
        and the rest are used for training.

        Parameters:
        percent: float, percentage of tasks to leave out as test (0 < percent < 1)
        random_order: bool, whether to randomize the task order
        random_state: int, seed for reproducibility

        Yields:
        (train_tasks, test_tasks, train_task_names, test_task_names): tuple
        """
        import random
        from math import ceil

        task_names = list(self.tasks.keys())
        n_total = len(task_names)
        n_test = max(1, ceil(percent * n_total))

        if random_order:
            rng = random.Random(random_state)
            rng.shuffle(task_names)

        for i in range(0, n_total, n_test):
            test_task_names = task_names[i:i + n_test]
            train_task_names = [name for name in task_names if name not in test_task_names]

            train_tasks = {name: self.tasks[name] for name in train_task_names}
            test_tasks = {name: self.tasks[name] for name in test_task_names}

            yield train_tasks, test_tasks, train_task_names, test_task_names

    def get_xy_split(self, tasks_dict):
        """
        Helper function to extract X and y from tasks dictionary
        
        Parameters:
        tasks_dict: dict, dictionary of tasks
        
        Returns:
        X: pd.DataFrame, features
        y: pd.Series, target (if specified)
        """
        # Concatenate all task data
        all_data = pd.concat(tasks_dict.values(), ignore_index=True)
            
        if self.target: 
            X = all_data.drop(columns=[self.target])
            columns = X.columns 
            X = X.values
            y = all_data[self.target].values.reshape(-1, 1)
            return X, y, columns
        else:
            return all_data, None
        
    def plot_corr(self):
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        
        # sns.heatmap(self.data.corr(), annot=True)
        # plt.show()

        plt.figure(figsize=(15, 12))  # Adjust dimensions as needed
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_corr_tasks(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        task_division = self.task_division
        for task_key, task_data in self.tasks.items():
            df = pd.DataFrame(task_data)
            plt.figure(figsize=(15, 12))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            title = ' '.join(['_'.join([task_division[i], str(task_key[i])]) for i in range(len(task_division))])
            plt.title(title)
            plt.tight_layout()
            plt.show()

    def plot_corr_tasks_stat(self, startcol=0, endcol=10):
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        import math 

        columns = tuple(set(self.data.columns) - set(self.task_division))
        columns = columns[startcol:endcol]

        corr_map = {}
        for column in columns:
            corr_map[column] = np.zeros(len(self.tasks))

        task_division = self.task_division
        
        target = self.target 
        for i, task_info in enumerate(self.tasks.items()):
            corr = task_info[1].corr()
            for column in columns:
                print(column, target)
                try:
                    corr_map[column][i] = corr.loc[column, target]
                except:
                    corr_map[column][i] = 0

        ncols = 3 
        nrows = math.ceil(len(columns)/ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*nrows, 4*ncols))
        axes = axes.flatten()
        
        titles = ['_'.join(f"{col}_{val}" for col, val in zip(task_division, task_key)) for task_key in self.tasks]    
        for i, corr_info in enumerate(corr_map.items()):
            axes[i].barh(titles, corr_info[1])
            axes[i].set_title(str(corr_info[0]))

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def plot_target_histograms(self, bins=10):
        """
        Plot histograms of the target feature for each task, with a vertical line for the mean.

        Parameters:
        bins: int, number of bins for the histogram
        """
        import matplotlib.pyplot as plt
        import math

        ncols = 2
        nrows = math.ceil(len(self.tasks) / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, (task_name, task_data) in enumerate(self.tasks.items()):
            if self.target in task_data.columns:
                target_values = task_data[self.target].dropna()
                mean_val = target_values.mean()

                axes[i].hist(target_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
                axes[i].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[i].set_title(f"Task: {task_name}")
                axes[i].set_xlabel(self.target)
                axes[i].set_ylabel("Frequency")
                axes[i].legend()
            else:
                axes[i].set_visible(False)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        
    def plot_feature_target_grid(self):
        """
        Plot pairwise relationships between features and the target.
        Uses seaborn.pairplot.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Combine numerical + target
        all_features = self.numerical_features + [self.target]

        # Filter only existing and numeric columns
        numeric_features = [f for f in all_features if f in self.data.columns]
        df_plot = self.data[numeric_features].copy()

        # Optional: apply log1p to target
        df_plot[self.target] = np.log1p(df_plot[self.target])

        # Drop NaNs to avoid plotting issues
        df_plot.dropna(inplace=True)

        # Pairplot
        sns.pairplot(df_plot, corner=True, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 1}, size=0.6)

        plt.suptitle("Pairwise Feature Relationships", fontsize=16, y=1.02)
        plt.show()
