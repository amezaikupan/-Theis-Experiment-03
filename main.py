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
from method import SGreedy, Pooling, Mean, SHat, CLF_Pool, Mode, Pooling_RF, Pooling_LGBM, Pooling_NN
from experiment import Experiment
import matplotlib.pyplot as plt 
np.random.seed(1234)
import visualize
 
# ************** For Life Expectancy data *****************
file_name = 'kc_house_data'
k = 10
mode = 0

data = pd.read_csv(f'{file_name}.csv', index_col=False).drop(columns=['sqft_living15', 'sqft_lot15', 'date', 'id'])
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'waterfront']
categorical_features = []
task_division = ['zipcode']
target = 'price'

# Filter populous zipcode 
top_zip_codes = [98103, 98038, 98115, 98052, 98117, 98042, 98034, 98118, 98023, 98006, 98133, 98059, 98058, 98155, 98074]
data = data[data['zipcode'].isin(top_zip_codes)]
data['price'] = data['price']/1000
with DataProcessor(data=data, task_division=task_division, target=target, numerical_features=numerical_features, categorical_features=categorical_features) as dataset:
        # dataset.plot_corr()
        # dataset.plot_corr_tasks_stat()
        # dataset.plot_corr_tasks()
        dataset.train_test_split()
# 
        pooling = Pooling()
        pooling_non_lin_1 = Pooling_RF()
        pooling_non_lin_2 = Pooling_LGBM()
        mean = Mean()
        sgreedy = SGreedy().set_params({'alpha': 0.001, 'use_hsic': True})
        shat = SHat().set_params({'alpha': 0.001, 'use_hsic': True})
        # nn = Pooling_NN()
# # 
# # 
        methods = [pooling,pooling_non_lin_1,pooling_non_lin_2, shat, sgreedy, mean]
        # methods = [mean]
        # methods = [pooling]
        experiment = Experiment(dataset, methods)

        file_annot = f"{file_name}_tasks_{dataset.n_tasks}_k_{k}_mode_{mode}"
        experiment.run_experiment(mode=mode, annot=f"{file_annot}", k=k) 
        # 
        if mode == 0:
                visualize._save_visualization_bar(annot=f"{file_annot}", k=k)
        else:
                visualize._save_visualization_line(annot=f"{file_annot}")

        # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot)
        # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98059,)')
        # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98058,)')
        # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98155,)')
        # # visualize.plot_per_sample_results(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', focus_task='(98006,)')
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='pooling', point_size=10)
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='random_forest', point_size=10)
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='lightGBM', point_size=10)
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='shat', point_size=10)
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='sgreedy', point_size=10)
        visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='mean', point_size=10)
        # visualize.plot_scatter_on_all_tasks(csv_path=f"per_sample_results/{file_annot}.csv", annot=file_annot, focus_method='neural-network', point_size=10)



# 
# 