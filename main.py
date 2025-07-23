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
from method import SGreedy, Pooling, Mean, SHat, CLF_Pool, Mode, Pooling_RF, Pooling_LGBM, Pooling_NN, Pooling_poly, SHat_RF, SHat_poly, SHat_GAM, Pooling_GAM, SGreedy_RF, SHat_LGBM, SGreedy_LGBM, SHat_poly,SGreedy_poly, SGreedy_GAM
from experiment import Experiment
import matplotlib.pyplot as plt 
np.random.seed(1234)
import visualize

# ************** For Life Expectancy data *****************
file_name = 'kc_house_data'
k = 10

# Mode:
#       - 0: Train on k domains, test on each of the rest.
#       - 1: Leave-one-out train 
mode = 0

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
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'waterfront','sqft_living15', 'sqft_lot15', 'house_age', 'renovation_age']#, 'lat', 'long']
categorical_features = []
task_division = ['zipcode']
target = 'price'
data = data[data['house_age'] >= 0]

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
        mean = Mean()

        pooling = Pooling()
        shat = SHat().set_params({'alpha': 0.001, 'use_hsic': False})
        sgreedy = SGreedy().set_params({'alpha': 0.001, 'use_hsic': False})

        pooling_rf = Pooling_RF()
        shat_rf = SHat_RF().set_params({'alpha': 0.001, 'use_hsic': False})
        sgreedy_rf = SGreedy_RF().set_params({'alpha': 0.001, 'use_hsic': False})


        pooling_lgbm = Pooling_LGBM()
        shat_lgbm = SHat_LGBM().set_params({'delta': 0.001, 'use_hsic': False})
        sgreedy_lgbm = SGreedy_LGBM().set_params({'delta': 0.001, 'use_hsic': False})

        pooling_poly = Pooling_poly().set_params(degree=2)
        shat_poly = SHat_poly().set_params(degree=2)
        sgreedy_poly = SGreedy_poly().set_params(degree=2)

        pooling_gam = Pooling_GAM()
        shat_GAM = SHat_GAM().set_params({'delta': 0.001, 'use_hsic': False})
        sgreedy_GAM = SGreedy_GAM().set_params({'delta': 0.001, 'use_hsic': False})
        
        methods = [mean, pooling, shat, sgreedy,
                    pooling_rf, shat_rf, sgreedy_rf, 
                    pooling_lgbm,shat_lgbm, sgreedy_lgbm, 
                    pooling_poly, shat_poly, sgreedy_poly, 
                    pooling_gam, shat_GAM, sgreedy_GAM]
   
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