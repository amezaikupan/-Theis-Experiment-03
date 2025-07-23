
import numpy as np
from sklearn import linear_model
from lightgbm import LGBMRegressor
from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import re_subset_search as subset_search 
from scipy import stats

 # Assuming subset_search is a custom module for model fitting and evaluation

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
    

## ********** POOLING ************

class Pooling(Method):
    def __init__(self, name='pooling'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        import matplotlib.pyplot as plt

        model = GradientBoostingRegressor().fit(X_train, y_train)
        PartialDependenceDisplay.from_estimator(model, X_train, features=[0, 1, 2,3,4,5,6,7])
        plt.show()

        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X) - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X) - y)**2
    
    def evaluate(self, X_test, y_test):
        return np.sqrt(np.mean((self.model.predict(X_test) - y_test)**2))
    
# Shat
class SHat(Method): 
    def __init__(self, name='shat'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):

        if self.params['mycode']:
            self.lasso_mask = None
            if (len(X_train[1]) < 13): 
                print('---No Lasso ---')
                s_hat = subset_search.full_search(
                    X_train, y_train, 
                    alpha=self.params['delta'],
                    valid_split=self.params['valid_split'],
                    n_samples_per_task=params['n_samples_per_task'], 
                    use_hsic=self.params['use_hsic']
                )
            else:
                print("--Use Lasso--")
                lasso_mask = lasso_binary_search_alpha(X_train, y_train)
                self.lasso_mask = lasso_mask
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
                    n_samples_per_task=params['n_samples_per_task'], 
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
                    n_samples_per_task=params['n_samples_per_task'], 
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
            
            self.model = model  
            self.selected_features = s_hat  # Store selected features
        else:
            self.selected_features = None

            if(self.params['is_classification_task']):
                self.mode = stats.mode(y_train)[0]
            else:
                self.mean = np.mean(y_train)
    
    def predict(self, X):
        # Implement evaluation logic for SGreedy        
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        return self.model.predict(X[:, self.selected_features])
    
    def cal_residuals_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        return self.model.predict(X[:, self.selected_features]) - y
    
    def cal_loss_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        return (self.model.predict(X[:, self.selected_features]) - y)**2
    
    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for SGreedy        
        if self.lasso_mask is not None:
            X_test = X_test[:, self.lasso_mask]

        if self.selected_features is not None:
            return np.sqrt(mse(self.model, X_test[:, self.selected_features], y_test))
        else:
            return np.mean((self.mean - y_test)**2)

# Sgreedy
class SGreedy(Method):
    def __init__(self, name='sgreedy'):
        super().__init__(name)
        self.params = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'is_classification_task': False, 
            'mycode': True,
        }
        if params is not None:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):

        if self.params['mycode']:
            s_greedy = subset_search.greedy_search(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task']
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

        self.model = model 
        self.selected_features = s_greedy
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
            return np.sqrt(mse(self.model, X_test[:, self.selected_features], y_test))
        
#  ****************** RANDOM FOREST *********************

# Pooling
class Pooling_RF(Method):
    def __init__(self, name='random_forest'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):

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
        prediction = self.model.predict(X)[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X)[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):

        prediction = self.model.predict(X_test)[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))

# Shat
class SHat_RF(Method): 
    def __init__(self, name='shat-rf'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):

        self.lasso_mask = None
        if (len(X_train[1]) < 13): 
            print('---No Lasso ---')
            s_hat = subset_search.full_search_rf(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic']
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
            )

        if(len(s_hat) != 0):
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train[:, s_hat], y_train)
            
            self.model = model 
            self.selected_features = s_hat
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)

    
    def predict(self, X):
        if self.lasso_mask is not None: 
            X = X[:, self.lasso_mask]
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        if self.lasso_mask is not None: 
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        if self.lasso_mask is not None: 
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        if self.lasso_mask is not None: 
            X_test = X_test[:, self.lasso_mask]

        prediction = self.model.predict(X_test[:, self.selected_features])[:, np.newaxis]
        return np.sqrt(np.mean((prediction - y_test)**2))

# Greedy
class SGreedy_RF(Method):
    def __init__(self, name='sgreedy-rf'):
        super().__init__(name)
        self.params = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'is_classification_task': False, 
            'mycode': True,
        }

        if params is not None:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params

        self.params = merged_params
        return self
    
    def fit(self, X_train, y_train, params):

        if self.params['mycode']:
            s_greedy = subset_search.greedy_search_rf(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task']
                # is_classification_task=self.params['is_classification_task']
            )
        else:
            s_greedy = subset_search.greedy_search_rf(
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
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train[:, s_greedy], y_train)

        self.model = model
        self.selected_features = s_greedy 
        return self
    
    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])
    
    def cal_residuals_list(self, X, y):
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis] - y
    
    def cal_loss_list(self, X, y):
        return (self.model.predict(X[:, self.selected_features]) - y)**2

    def evaluate(self, X_test, y_test):
        if(self.params['is_classification_task']):
            return accuracy_score(y_test, self.model.predict(X_test[:, self.selected_features]))
        else:
            # return mae(self.model, X_test[:, self.selected_features], y_test)
            # return mse_2(np.expm1(self.model.predict(X_test[:, self.selected_features])), y_test)
            return np.sqrt(mse(self.model, X_test[:, self.selected_features], y_test))


#  ****************** LGBM **************************

# Pooling
class Pooling_LGBM(Method):
    def __init__(self, name='lightGBM'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        self.model = LGBMRegressor(
            objective='regression', 
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31
        )
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        prediction = self.model.predict(X)[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X)[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        prediction = self.model.predict(X_test)[:, np.newaxis]

        return np.sqrt(np.mean((prediction - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

# Shat
class SHat_LGBM(Method): 
    def __init__(self, name='shat-LGBM'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        self.model = None
        self.lasso_mask = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
            'mycode': True
        }

        self.params = {**default_params, **(params or {})}
        return self
    
    def fit(self, X_train, y_train, params):
        if X_train.shape[1] < 13:
            print('---No Lasso ---')
            s_hat = subset_search.full_search_lgbm(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic']
            )
        else:
            print("--Use Lasso--")
            self.lasso_mask = lasso_binary_search_alpha(X_train, y_train)
            print(f"Lasso Mask: {self.lasso_mask}")
            X_train = X_train[:, self.lasso_mask]

            s_hat = subset_search.full_search_lgbm(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic']
            )

        if len(s_hat) != 0:
            self.model = LGBMRegressor(
                objective='regression', 
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31
            )
            self.model.fit(X_train[:, s_hat], y_train)
            self.selected_features = s_hat
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)

    def predict(self, X):
        if self.selected_features is None:
            return np.full((X.shape[0], 1), self.mean)
        
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]

        pred = self.model.predict(X[:, self.selected_features])
        return pred[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        if self.lasso_mask is not None: 
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]
        return prediction - y

    def cal_loss_list(self, X, y):
        if self.lasso_mask is not None: 
            X = X[:, self.lasso_mask]
        pred = self.model.predict(X[:, self.selected_features])
        return (pred - y)**2

    def evaluate(self, X_test, y_test):
        if self.lasso_mask is not None: 
            X_test = X_test[:, self.lasso_mask]
        pred = self.model.predict(X_test[:, self.selected_features])
        return np.sqrt(np.mean((pred - y_test)**2))
    
# Greedy
class SGreedy_LGBM(Method): 
    def __init__(self, name='sgreedy-LGBM'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        self.model = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'mycode': True
        }

        self.params = {**default_params, **(params or {})}
        return self
    
    def fit(self, X_train, y_train, params):
        s_greedy = subset_search.greedy_search_lgbm(
            X_train, y_train, 
            alpha=self.params['delta'],
            valid_split=self.params['valid_split'],
            n_samples_per_task=params['n_samples_per_task'], 
            use_hsic=self.params['use_hsic']
        )
        
        if len(s_greedy) != 0:
            # self.model = LGBMRegressor()
            self.model = LGBMRegressor(
                objective='regression', 
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31
            )
            self.model.fit(X_train[:, s_greedy], y_train)
            self.selected_features = s_greedy
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)

    def predict(self, X):
        if self.selected_features is None:
            return np.full((X.shape[0], 1), self.mean)
        
        pred = self.model.predict(X[:, self.selected_features])
        return pred[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])[:, np.newaxis]
        return prediction - y

    def cal_loss_list(self, X, y):
        pred = self.model.predict(X[:, self.selected_features])
        return (pred - y)**2

    def evaluate(self, X_test, y_test):
        pred = self.model.predict(X_test[:, self.selected_features])
        return np.sqrt(np.mean((pred - y_test)**2))
    
# ******************* POLYNOMIAL *******************

# Pooling
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
        return np.sqrt(np.mean((self.model.predict(X_test) - y_test)**2))
        # return np.sqrt(mse(self.model, X_test, y_test))

# shat
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
        
        self.lasso_mask = None
        if (len(X_train[1]) < 13): 
            print('---No Lasso ---')
            s_hat = subset_search.full_search_poly(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'], 
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

        print(f"aboutS_hay poly found {s_hat}")

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
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        if self.lasso_mask is not None:
            X_test = X_test[:, self.lasso_mask]
        prediction = self.model.predict(X_test[:, self.selected_features])#[:, np.newaxis]
        return np.sqrt(np.mean((prediction - y_test)**2))

# Sgreedy
class SGreedy_poly(Method): 
    def __init__(self, name='sgreedy-poly'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, degree=2, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
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
        
        s_greedy = subset_search.greedy_search_poly(
            X_train, y_train, 
            alpha=self.params['delta'],
            valid_split=self.params['valid_split'],
            n_samples_per_task=params['n_samples_per_task'], 
            degree=self.degree,
            use_hsic=self.params['use_hsic']
            # is_classification_task = self.params['is_classification_task']
        )

        if(len(s_greedy) != 0):
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            self.selected_features = s_greedy
            self.model = make_pipeline(PolynomialFeatures(self.degree), linear_model.LinearRegression())

            self.model.fit(X_train[:, self.selected_features], y_train)
            return self 
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)
   
  
    def predict(self, X):
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]
        return prediction - y
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])#[:, np.newaxis]
        return (prediction - y)**2
 
    def evaluate(self, X_test, y_test):
        prediction = self.model.predict(X_test[:, self.selected_features])#[:, np.newaxis]
        return np.sqrt(np.mean((prediction - y_test)**2))


#  *************** GAM ******************

# Pooling
from pygam import LinearGAM, s
class Pooling_GAM(Method):
    def __init__(self, name='Pooling_GAM'):
        super().__init__(name)
        self.model = None
        self.terms = None

    def set_params(self, max_features=None):
        self.max_features = max_features
        return self

    
    def fit(self, X_train, y_train, params=None):
        n_features = X_train.shape[1]

        self.terms = sum([s(i) for i in range(n_features)], start=s(0))

        self.model = LinearGAM(self.terms)
        self.model.fit(X_train, y_train.flatten())

        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def cal_residuals_list(self, X, y):
        return self.predict(X) - y.flatten()

    def cal_loss_list(self, X, y):
        return (self.predict(X) - y.flatten())**2

    def evaluate(self, X_test, y_test):
        residuals = self.predict(X_test) - y_test.flatten()
        return np.sqrt(np.mean(residuals**2))  # RMSE
    

# shat
class SHat_GAM(Method): 
    def __init__(self, name='shat-gam'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
        }

        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self

    def fit(self, X_train, y_train, params):
        self.lasso_mask = None
        if X_train.shape[1] < 13:
            print('---No Lasso ---')
            s_hat = subset_search.full_search_gam(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'],
                use_hsic=self.params['use_hsic']
            )
        else:
            print("--Use Lasso--")
            lasso_mask = lasso_binary_search_alpha(X_train, y_train)
            self.lasso_mask = lasso_mask
            X_train = X_train[:, lasso_mask]

            s_hat = subset_search.full_search_gam(
                X_train, y_train, 
                alpha=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task=params['n_samples_per_task'],
                use_hsic=self.params['use_hsic']
            )

        print(f"S_hat GAM found {s_hat}")

        if len(s_hat) != 0:
            self.selected_features = s_hat
            gam_terms = s(0)
            for j in range(1, len(s_hat)):
                gam_terms += s(j)

            # gam_terms = sum([s(i) for i in range(len(s_hat))])
            self.model = LinearGAM(gam_terms)
            self.model.fit(X_train[:, s_hat], y_train)
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)

        return self

    def predict(self, X):
        if self.selected_features is None:
            return np.full((X.shape[0], 1), self.mean)
        
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])
        return prediction - y.ravel()
    
    def cal_loss_list(self, X, y):
        if self.lasso_mask is not None:
            X = X[:, self.lasso_mask]
        prediction = self.model.predict(X[:, self.selected_features])
        return (prediction - y)**2

    def evaluate(self, X_test, y_test):
        if self.lasso_mask is not None:
            X_test = X_test[:, self.lasso_mask]
        prediction = self.model.predict(X_test[:, self.selected_features])
        return np.sqrt(np.mean((prediction - y_test)**2))
    

# Sgreedy
class SGreedy_GAM(Method): 
    def __init__(self, name='sgreedy-gam'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.4,
            'use_hsic': False, 
            'is_classification_task': False, 
        }

        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self

    def fit(self, X_train, y_train, params):
        
        s_greedy = subset_search.greedy_search_gam(  # You will define this
            X_train, y_train, 
            alpha=self.params['delta'],
            valid_split=self.params['valid_split'],
            n_samples_per_task=params['n_samples_per_task'],
            use_hsic=self.params['use_hsic']
        )

        print(f"S_hat GAM found {s_greedy}")

        if len(s_greedy) != 0:
            self.selected_features = s_greedy
            gam_terms = s(0)
            for j in range(1, len(s_greedy)):
                gam_terms += s(j)

            # gam_terms = sum([s(i) for i in range(len(s_hat))])
            self.model = LinearGAM(gam_terms)
            self.model.fit(X_train[:, s_greedy], y_train)
        else:
            self.selected_features = None
            self.mean = np.mean(y_train)
        return self

    def predict(self, X):
        if self.selected_features is None:
            return np.full((X.shape[0], 1), self.mean)
        return self.model.predict(X[:, self.selected_features])[:, np.newaxis]
    
    def cal_residuals_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])
        return prediction - y.ravel()
    
    def cal_loss_list(self, X, y):
        prediction = self.model.predict(X[:, self.selected_features])
        return (prediction - y)**2

    def evaluate(self, X_test, y_test):
        prediction = self.model.predict(X_test[:, self.selected_features])
        return np.sqrt(np.mean((prediction - y_test)**2))
    


# ***** NN ************

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
