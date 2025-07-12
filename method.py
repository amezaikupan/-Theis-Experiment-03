
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor 
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import accuracy_score

import os 
import sys 
import re_subset_search as subset_search  # Assuming subset_search is a custom module for model fitting and evaluation

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *  # Assuming mse is a custom utility function for evaluation metrics

class Method:
    def __init__(self, name):
        self.name = name
    def  fit(self, X_train, y_train, params):
        """
        Fit the model to the training data.
        This is a placeholder for actual fitting logic.
        """
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the given data.
        This is a placeholder for actual evaluation logic.
        """
        # Return a dummy score for demonstration purposes
        return np.random.rand()
    
    def cal_residuals_list(self, X, y):
        return np.random.rand()
    
    def cal_loss_list(self,X,y):
        return np.random.rand()
    
    def predict(self, X):
        return np.random.rand()
    
class SGreedy(Method):
    def __init__(self, name='sgreedy'):
        super().__init__(name)
        self.params = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.6,
            'is_classification_task': False, 
            'mycode': True,
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        # y_train = np.log1p(y_train)  # Log transform the target variable

        if self.params['mycode']:
            s_greedy = subset_search.greedy_search(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task']
                # is_classification_task=self.params['is_classification_task']
            )
        else:
            s_greedy = subset_search.greedy_subset(
                X_train, y_train, 
                delta=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_ex=params['n_samples_per_task']
                # is_classification_task=self.params['is_classification_task']
            )

        if(self.params['is_classification_task']):
            model = linear_model.LogisticRegression()
            model.fit(X_train[:, s_greedy], y_train)
        else:
            model = linear_model.LinearRegression()
            model.fit(X_train[:, s_greedy], y_train)

        # print("Model coefficient", model.coef_)
        self.model = model  # Store the fitted model
        self.selected_features = s_greedy  # Store selected features
        return self
    
    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X[:, self.selected_features]) - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X[:, self.selected_features]) - y)**2

    def evaluate(self, X_test, y_test):
        if(self.params['is_classification_task']):
            return accuracy_score(y_test, self.model.predict(X_test[:, self.selected_features]))
        else:
            # return mae(self.model, X_test[:, self.selected_features], y_test)
            # return mse_2(np.expm1(self.model.predict(X_test[:, self.selected_features])), y_test)
            return np.sqrt(mse(self.model, X_test[:, self.selected_features], y_test))

class SHat(Method): 
    def __init__(self, name='shat'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.6,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        # y_train = np.log1p(y_train)

        if self.params['mycode']:
            self.lasso_mask = None
            if (len(X_train[1]) < 13): 
                print('---No Lasso ---')
                s_hat = subset_search.full_search(
                    X_train, y_train, 
                    alpha=self.params['delta'],
                    valid_split=self.params['valid_split'],
                    n_samples_per_task_list=params['n_samples_per_task'], 
                    use_hsic=self.params['use_hsic']
                    # is_classification_task = self.params['is_classification_task']
                )
            else:
                print("--Use Lasso--")
                lasso_mask = lasso_binary_search_alpha(X_train, y_train)
                self.lasso_mask = lasso_mask
                print(f"Lasso Mask: {lasso_mask}")
                X_train = X_train[:, lasso_mask]
                
                s_hat = subset_search.full_search(
                    X_train, y_train, 
                    alpha=self.params['delta'],
                    valid_split=self.params['valid_split'],
                    n_samples_per_task=params['n_samples_per_task'], 
                    use_hsic=self.params['use_hsic'],
                    # is_classification_task = self.params['is_classification_task']
                )

        else:
            self.lasso_mask = None
            if (len(X_train[1]) < 13): 
                print('---No Lasso ---')
                s_hat = subset_search.subset(
                    X_train, y_train, 
                    delta=self.params['delta'],
                    valid_split=self.params['valid_split'],
                    n_samples_per_task_list=params['n_samples_per_task'], 
                    use_hsic=self.params['use_hsic']
                    # is_classification_task = self.params['is_classification_task']
                )
            else:
                print("--Use Lasso--")
                lasso_mask = lasso_binary_search_alpha(X_train, y_train)
                self.lasso_mask = lasso_mask

                X_train = X_train[:, lasso_mask]
                
                s_hat = subset_search.full_search(
                    X_train, y_train, 
                    delta=self.params['delta'],
                    valid_split=self.params['valid_split'],
                    n_samples_per_task_list=params['n_samples_per_task'], 
                    use_hsic=self.params['use_hsic'],
                    # is_classification_task = self.params['is_classification_task']
                )


        if(len(s_hat) != 0):
            if(self.params['is_classification_task']):
                model = linear_model.LogisticRegression()
                model.fit(X_train[:, s_hat], y_train)
            else:
                model = linear_model.LinearRegression()
                model.fit(X_train[:, s_hat], y_train)
            
            self.model = model  # Store the fitted model
            # print(model.coef_)
            self.selected_features = s_hat  # Store selected features
        else:
            self.selected_features = None
            from scipy import stats
            if(self.params['is_classification_task']):
                self.mode = stats.mode(y_train)[0]
            else:
                self.mean = np.mean(y_train)
    
    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X[:, self.selected_features]) - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X[:, self.selected_features]) - y)**2
    
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for SGreedy        
        if self.lasso_mask is not None:
            X_test = X_test[:, self.lasso_mask]

        if(self.params['is_classification_task']):
            if self.selected_features is not None:
                print("Evaluate selected features")
                return accuracy_score(y_test, self.model.predict(X_test[:, self.selected_features]))
            else:
                return accuracy_score(y_test, np.full_like(y_test, self.mode))
        else:
            if self.selected_features is not None:
                # return mse_2(np.expm1(self.model.predict(X_test[:, self.selected_features])),y_test)
                # return mae(self.model, X_test[:, self.selected_features], y_test)
                return np.sqrt(mse(self.model, X_test[:, self.selected_features], y_test))
            else:
                return np.mean((self.mean - y_test)**2)
                # return np.mean((y_test, self.model.predict(X_test[:,  self.selected_features]))**2)
    

class SHat_RF(Method): 
    def __init__(self, name='shat-RF'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.6,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        # y_train = np.log1p(y_train)

        self.lasso_mask = None
        if (len(X_train[1]) < 13): 
            print('---No Lasso ---')
            s_hat = subset_search.full_search_rf(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task_list=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic']
                # is_classification_task = self.params['is_classification_task']
            )
        else:
            print("--Use Lasso--")
            lasso_mask = lasso_binary_search_alpha(X_train, y_train)
            self.lasso_mask = lasso_mask
            print(f"Lasso Mask: {lasso_mask}")
            X_train = X_train[:, lasso_mask]
            
            s_hat = subset_search.full_search_rf(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic'],
                # is_classification_task = self.params['is_classification_task']
            )

        if(len(s_hat) != 0):
            model = RandomForestRegressor()
            model.fit(X_train[:, s_hat], y_train)
            
            self.model = model  # Store the fitted model
            # print(model.coef_)
            self.selected_features = s_hat  # Store selected features
        else:
            self.selected_features = None
            from scipy import stats
            self.mean = np.mean(y_train)

    
    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        # print(f"Shape y{y.shape}, X{X.shape}")
        # print(f"Model predict", self.model.predict(X).shape)[:, :np.newaxis]
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]

        # prediction = prediction[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        prediction = self.model.predict(X_test[:, self.selected_features])[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

class SHat_poly(Method): 
    def __init__(self, name='shat-poly'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, degree=2, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        self.degree = degree
        return self
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        # y_train = np.log1p(y_train)

        self.lasso_mask = None
        if (len(X_train[1]) < 13): 
            print('---No Lasso ---')
            s_hat = subset_search.full_search_poly(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task_list=params['n_samples_per_task'], 
                degree=self.degree,
                use_hsic=self.params['use_hsic']
                # is_classification_task = self.params['is_classification_task']
            )
        else:
            print("--Use Lasso--")
            lasso_mask = lasso_binary_search_alpha(X_train, y_train)
            self.lasso_mask = lasso_mask
            print(f"Lasso Mask: {lasso_mask}")
            X_train = X_train[:, lasso_mask]
            
            s_hat = subset_search.full_search_poly(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
                degree=self.degree,
                use_hsic=self.params['use_hsic'],
                # is_classification_task = self.params['is_classification_task']
            )

        print(f"S_hay poly found {s_hat}")

        if(len(s_hat) != 0):
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            self.selected_features = s_hat
            self.model = make_pipeline(PolynomialFeatures(self.degree), linear_model.LinearRegression())

            self.model.fit(X_train[:, self.selected_features], y_train)
            return self 
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)
   
  
    def predict(self, X):
        X_selected = X[:, self.selected_features]
        print(X_selected[0])
        predictions = self.model.predict(X_selected).squeeze()
        outlier_id = predictions < -2000
        print(np.sum(outlier_id))
        print(self.selected_features)
        print(X[:, self.selected_features][outlier_id])
        # ldsjglkj
   
        # lskdjg
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        # print(f"Shape y{y.shape}, X{X.shape}")
        # print(f"Model predict", self.model.predict(X).shape)[:, :np.newaxis]
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]

        # prediction = prediction[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        prediction = self.model.predict(X_test[:, self.selected_features])#[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

class Pooling(Method):
    def __init__(self, name='pooling'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Pooling
        # This is a placeholder for actual fitting logic
        # y_train = np.log1p(y_train)  # Log transform the target variable
        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)
       
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X) - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X) - y)**2
    
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        return np.sqrt(np.mean((self.model.predict(X_test) - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

class Pooling_poly(Method):
    def __init__(self, name='polynomial'):
        super().__init__(name)
        self.degree = None

    def set_params(self, degree=2):
        self.degree = degree
        self.name = ' '.join([self.name,str(degree)])
        return self
    
    def fit(self, X_train, y_train, params):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        
        self.model = make_pipeline(PolynomialFeatures(self.degree), linear_model.LinearRegression())
        self.model.fit(X_train, y_train)
        return self 
    def predict(self, X):
        return self.model.predict(X)
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X) - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X) - y)**2
    
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        return np.sqrt(np.mean((self.model.predict(X_test) - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

class Pooling_RF(Method):
    def __init__(self, name='random_forest'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Pooling
        # This is a placeholder for actual fitting logic
        # y_train = np.log1p(y_train)  # Log transform the target variable

        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self
    

    def predict(self, X):
        return self.model.predict(X)[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        # print(f"Shape y{y.shape}, X{X.shape}")
        # print(f"Model predict", self.model.predict(X).shape)[:, :np.newaxis]
        prediction = self.model.predict(X)[:, np.newaxis]

        # prediction = prediction[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X)[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        # return mae(self.model, X_test, y_test)
        # return mse_2(np.expm1(self.model.predict(X_test)), y_test)
        prediction = self.model.predict(X_test)[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))
    
class Pooling_LGBM(Method):
    def __init__(self, name='lightGBM'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Pooling
        # This is a placeholder for actual fitting logic

        # y_train = np.log1p(y_train)  # Log transform the target variable
        self.model = LGBMRegressor(
            objective='regression',   # or 'regression_l1', 'huber', etc.
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31
        )
        self.model.fit(X_train, y_train)
        return self
    

    def predict(self, X):
        return self.model.predict(X)[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        # print(f"Shape y{y.shape}, X{X.shape}")
        # print(f"Model predict", self.model.predict(X).shape)[:, :np.newaxis]
        prediction = self.model.predict(X)[:, np.newaxis]

        # prediction = prediction[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X)[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        # return mae(self.model, X_test, y_test)
        # return mse_2(np.expm1(self.model.predict(X_test)), y_test)
        prediction = self.model.predict(X_test)[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

class Pooling_NN(Method):
    def __init__(self, name="neural-network"):
        super().__init__(name)

    def fit(self, X_train, y_train, params):

        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.model.fit(X_train, y_train)
        return self 
    
    def predict(self, X):
        return self.predict(X)
    
    def cal_residuals_list(self, X, y):
        prediction = self.model.predict(X)
        prediction = prediction[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X) - y)**2
    
    def evaluate(self, X_test, y_test):
        return np.sqrt(mse(self.model, X_test, y_test))

class Mean(Method):
    def __init__(self, name='mean'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Mean
        self.mean = np.mean(y_train)
        return self

    def predict(self, X):
        return np.full(shape=(len(X), 1), fill_value=self.mean)
    def cal_residuals_list(self, X, y):
        return self.mean - y
    
    def cal_loss_list(self, X, y):
        return (self.mean - y)**2
    
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Mean
        # return mse_2(np.full_like(y_test, self.mean), y_test)
        return np.sqrt(np.mean(((self.mean - y_test)**2)))
    
class CLF_Pool(Method):
    def __init__(self, name='pool'):
        super().__init__(name)

    def fit(self, X_train, y_train, params):
        self.model = linear_model.LogisticRegression()
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        # return super().evaluate(X_test, y_test)
        return accuracy_score(y_test > 0.5, self.model.predict(X_test))

class Mode(Method):
    def __init__(self, name='mean'):
        super().__init__(name)

    def fit(self, X_train, y_train, params):
        from scipy import stats
        self.mode = stats.mode(y_train, keepdims=True)[0]

    def evaluate(self, X_test, y_test):
        return accuracy_score(y_test > 0.5, np.full_like(y_test, self.mode))
