# Things that need to be process:
# 1. Dvision of tasks (always categorical features)
# 2. Load and process of data 
# 3. Division of train and test data 
# 4. Model training and evaluation 

# TODO: Understand the MSE loss (And what s_greedy - alpha means?)
# TODO: Implement s_hat class too

import pandas as pd 
import numpy as np
from data_processor import DataProcessor
from method import SGreedy, Pooling, Mean, SHat, CLF_Pool, Mode, Pooling_RF, Pooling_LGBM, Pooling_NN, Pooling_poly, SHat_RF, SHat_poly
from experiment import Experiment
import matplotlib.pyplot as plt 
np.random.seed(1234)
import visualize

def clip_extremes(df, lower_pct=0.01, upper_pct=0.99):
    return df.clip(
        lower=df.quantile(lower_pct),
        upper=df.quantile(upper_pct),
        axis=1
    )

def remove_outliers_percentile(df, column, lower_pct=0.01, upper_pct=0.99):
    lower = df[column].quantile(lower_pct)
    upper = df[column].quantile(upper_pct)
    return df[(df[column] >= lower) & (df[column] <= upper)]
 
# ************** For Life Expectancy data *****************
file_name = 'kc_house_data'#_outliers_dect'
k = 10
mode = 0


column_mask = [2,3,9]

data = pd.read_csv(f'{file_name}.csv', index_col=False).drop(columns=['id', 'lat', 'long'])
# Convert date string to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S')

# Extract year of sale
data['yr_sold'] = data['date'].dt.year

# Calculate age of the house
data['house_age'] = data['yr_sold'] - data['yr_built']

# Calculate age since renovation, 0 if never renovated
data['renovation_age'] = data.apply(
    lambda row: row['yr_sold'] - row['yr_renovated'] if row['yr_renovated'] > 0 else 0,
    axis=1
)

data = data.drop(columns=['yr_sold', 'yr_built', 'yr_renovated', 'date'])
# print(len(data))

print("Before data leng", len(data))
# data = remove_outliers_percentile(data, column='sqft_basement')
# data = remove_outliers_percentile(data, column='sqft_lot')

# print("After clipping length", len(data))
# lasdkjg
print(data.columns)
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'waterfront','sqft_living15', 'sqft_lot15', 'house_age', 'renovation_age']#, 'lat', 'long']
categorical_features = []
task_division = ['zipcode']
target = 'price'
print(data['house_age'].value_counts())
data = data[data['house_age'] >= 0]

# data = data[data['bedrooms'] < 10]
# print(data['grade'].value_counts())

# hhh
# Filter populous zipcode 
top_zip_codes = [98103, 98038, 98115, 98052, 98117, 98042, 98034, 98118, 98023, 98006, 98133, 98059, 98058, 98155, 98074]
data = data[data['zipcode'].isin(top_zip_codes)]
data['price'] = data['price']/1000

with DataProcessor(data=data, task_division=task_division, target=target, numerical_features=numerical_features, categorical_features=categorical_features) as dataset:
        # dataset.plot_corr()
        # dataset.plot_corr_tasks_stat()
        # dataset.plot_corr_tasks()
        # dataset.plot_feature_target_grid()
        # dataset.data = dataset.data.iloc[:, column_mask]
        dataset.train_test_split()
        # print(dataset.data.iloc[0])
# 
        pooling = Pooling()
        pooling_non_lin_1 = Pooling_RF()
        pooling_non_lin_2 = Pooling_LGBM()
        pooling_poly = Pooling_poly().set_params(degree=2)
        mean = Mean()
        sgreedy = SGreedy().set_params({'alpha': 0.001, 'use_hsic': False})
        shat = SHat().set_params({'alpha': 0.001, 'use_hsic': False})
        shat_rf = SHat_RF().set_params({'alpha': 0.001, 'use_hsic': False})
        shat_poly = SHat_poly().set_params(degree=2, params={'alpha': 0.001, 'use_hsic': False})
        
        nn = Pooling_NN()
        print(pooling_poly.name)
# 
        # methods = [pooling,pooling_non_lin_1,pooling_non_lin_2, pooling_poly, shat, sgreedy, mean]
        # methods = [shat_poly, shat_rf, shat]
        methods = [shat_poly]
        # methods = [shat_poly]
        experiment = Experiment(dataset, methods)

        file_annot = f"{file_name}_tasks_{dataset.n_tasks}_k_{k}_mode_{mode}"
        experiment.run_experiment(mode=mode, annot=f"{file_annot}", k=k) 
        # 
        if mode == 0:
                visualize._save_visualization_bar(annot=f"{file_annot}", k=k)
        else:
                visualize._save_visualization_line(annot=f"{file_annot}")

        for method in methods:
                visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method=method.name, point_size=10, set='train')
                visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method=method.name, point_size=10, set='test')


# #         # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot)
# #         # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98059,)')
# #         # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98058,)')
# #         # # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98155,)')
# #         # # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98006,)')
    

# # # # 
# # # # 