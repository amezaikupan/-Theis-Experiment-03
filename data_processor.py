import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
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
        print('Before processing data:')
        print(np.sum(data.isna()))
        cols_removed = self.process_na_cols()
        self.numerical_features = [feat for feat in numerical_features if feat not in cols_removed]
        self.categorical_features = [col for col in self.data.columns
                if col not in numerical_features
                and col not in task_division
                and col != target]
        print("Columns removed due to high NA percentage:")
        print(cols_removed)
        self.process_na_rows()

        print('After processing data:')
        print(np.sum(self.data.isna()))
        print(f"Number of rows: {len(self.data)}")


        print('\n **** PROCESS CATEGORICAL & NUMERICAL COLUMNS **** ')
        data_features = self.numerical_features + self.categorical_features
        # self.data  = 
        
        # scaler = StandardScaler()
        # self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        print("Before processing categorical & numerical columns")
        print(f"Number of columns: {len(self.data.columns)}")
        self.process_features()
        print("After processing categorical & numerical columns")
        print(f"Number of columns: {len(self.data.columns)}")
        print(f"Nan value {np.sum(self.data.isna())}")

        # Remove outliners 
        self.clip_extreme_values()

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
            # print("TASK NAN", np.sum(tasks[name].isna()))
            # print("TASK", len(tasks[name].columns) )
            # print("TASK LENGTH", len(tasks[name]))
            # print("ROW WITH TASK NAN", tasks[name][tasks[name].isna().any(axis=1)])
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


    def clip_extreme_values(self, lower_pct=0, upper_pct=0.95):
        lower = self.data[self.target].quantile(lower_pct)
        upper = self.data[self.target].quantile(upper_pct)
        self.data = self.data[(self.data[self.target] >= lower) & (self.data[self.target] <= upper)]    

    def process_features(self):
        # PROCESS NUMERICAL FEATURES
        # Coerce to numeric with NaNs for non-convertibles
        numeric_data = self.data[self.numerical_features].apply(pd.to_numeric, errors='coerce')
        print("____________numeric data ______________")
        print(numeric_data)

        # print("PROCESS_FEATUREs", np.sum(numeric_data.isna()))
        # print(len(numeric_data))

        # Ensure values are >= -1 (log1p is undefined for values < -1)
        # for col in self.numerical_features:
        #     if (numeric_data[col] < -1).any():
        #         print(f"Skipping column {col}: contains values < -1")
        #         continue
        #     numeric_data[col] = np.log1p(numeric_data[col])

        scaler = StandardScaler()
        numeric_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns=self.numerical_features)      
        numeric_data.index = self.data.index
        
        # PROCESS CATEGORICAL FEATURES
        categorical_data = pd.get_dummies(self.data[self.categorical_features], drop_first=True)
        # print("PROCESS CATEGORICAL FEATURES", np.sum(categorical_data.isna()))
        # print(len(categorical_data))
        categorical_data.index = self.data.index

        task_division_data = self.data[self.task_division]        
        target_data = self.data[self.target]
        # MERGE TO DATA
        self.data = pd.concat([numeric_data, categorical_data, task_division_data, target_data], axis=1)
        print("^^^^^^^^^^^ process_features ^^^^^^^^^^^^^")
        print(self.data.head(1))
    
    def train_test_split(self, test_split=0.3, random_state=42):
        """
        Split tasks into train and test sets based on task division values.
        The latter (biggest combination values) go to test set.
        
        Parameters:
        test_split: float, proportion of tasks to use for testing
        random_state: int, random seed for reproducibility
        
        Returns:
        train_tasks: dict, training tasks
        test_tasks: dict, testing tasks
        """
        # Get all task names (keys) and sort them
        # This ensures consistent ordering for the "latter" tasks
        task_names = list(self.tasks.keys())
        
        # Sort task names to get consistent ordering
        # For tuples (Year, Gender), this will sort lexicographically
        # task_names_sorted = sorted(task_names)

        # Or randomize  
        import random
        task_names_sorted = task_names
        random.shuffle(task_names_sorted) 
        
        # Calculate split point
        n_tasks = len(task_names_sorted)
        n_test_tasks = max(1, int(n_tasks * test_split))  # Ensure at least 1 test task
        
        # Split: latter tasks go to test
        train_task_names = task_names_sorted[:-n_test_tasks]
        test_task_names = task_names_sorted[-n_test_tasks:]
        
        # Create train and test task dictionaries
        train_tasks = {name: self.tasks[name] for name in train_task_names}
        test_tasks = {name: self.tasks[name] for name in test_task_names}
        
        # print(f"Train tasks: {train_task_names}")
        # print(f"Test tasks: {test_task_names}")

        # print(train_tasks)

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
