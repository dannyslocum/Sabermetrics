import numpy as np
import pandas as pd
import zipfile
from sklearn.preprocessing import scale

'''
Data from Lahman's Baseball Database
http://www.seanlahman.com/baseball-archive/statistics/'
'''

zf = zipfile.ZipFile('lahman_data.zip')
df = pd.read_csv(zf.open('core/Teams.csv'))
df.describe(percentiles=[.05, .95])
df.year.value_counts(normalize=True)
np.random.seed(13)

"""
steps
1. preprocessing
    Fill in missing data
    ----------------------
            sklearn.impute.MissingIndicator
            sklearn.impute.SimpleImputer
            sklearn.impute.KNNImputer
            sklearn.impute.IterativeImputer
            pd.fillna()
    ----------------------
    

    Convert categorical to numerical
    --------------------------------
            sklearn.preprocessing.OrdinalEncoder
            sklearn.preprocessing.OneHotEncoder
            pd.get_dummies()
    --------------------------------


    Scale Data
    -------------------------
            sklearn.preprocessing.scale
            sklearn.preprocessing.MinMaxScaler
            sklearn.preprocessing.StandardScaler
            sklearn.preprocessing.RobustScaler
            sklearn.preprocessing.MaxAbsScaler
            sklearn.preprocessing.Normalizer
    ------------------------


    Reduce Dimensionality
    ------------------------
            sklearn.decomposition.PCA
            LDA?
        
    
    Other
    -------------------------
            sklearn.preprocessing.LabelBinarizer
            sklearn.preprocessing.LabelEncoder
            sklearn.preprocessing.MultiLabelBinarizer
            
            sklearn.preprocessing.KBinsDiscretizer
            
            sklearn.preprocessing.Binarizer
            
            sklearn.preprocessing.PolynomialFeatures
    -------------------------
    

2. choose and create model(s)
    Linear regression
    -----------------------
        sklearn.linear_model.LinearRegression
                desc: least squares regression
                ------- 
                
                ------- 
            
        sklearn.linear_model.Ridge 
                desc: least squares (L2 norm) but find smallest coefficients to be less susceptible to random noise
                ------- 
                    alpha (optional) <- float: scaling factor for penalty term
                ------- 
            
        sklearn.linear_model.RidgeCV
                desc: Ridge with Cross Validation
                ------- 
                    alphas <- List[float]: scaling factor for penalty term
                ------- 
                        
        sklearn.linear_model.BayesianRidge
                desc: Ridge but with bayesian techniques by making assumptions about probability distribution prior to fitting model
                ------- 
                    alpha_1 (optional) <- float: scaling factor for penalty term
                    alpha_2 (optional) <- float: scaling factor for penalty term
                    lambda_1 (optional) <- float: precision of the model's weights
                    lambda_2 (optional) <- float: precision of the model's weights
                ------- 
            
        sklearn.linear_model.Lasso 
                desc: sparse linear regression using L1 norm). Typically reduced some features to weights of 0
                ------- 
                    alpha (optional) <- float: scaling factor for penalty term
                ------- 
            
        sklearn.linear_model.LassoCV
                desc: Lasso with Cross Validation
                ------- 
                    alphas <- List[float]: scaling factor for penalty term
                ------- 
    -----------------------
    
    
    Linear classification
    -----------------------
        sklearn.linear_model.LogisticRegression
                ------- 
                solver (optional) <- str: select type of solver model
                    'ovr' as one-vs-rest binary classifier
                    'lbfgs'; 'liblinear','newton-cg','sag','saga' <-- multi-class
                multi-class (optional) <- str: 
                    'multinomal' to select multi-class
                max_iter (optional) <- int: set the maximum number of iterations the solver takes
                penalty (optional) <- str: specify the penalty method for weights
                    'l1' for L1 norm
                    'l2' for L2 norm
                -------    
            
        sklearn.linear_model.LogisticRegressionCV
                ------- 
                solver (optional) <- str: select type of solver model
                    'ovr' as one-vs-rest binary classifier
                    'lbfgs'; 'liblinear','newton-cg','sag','saga' <-- multi-class
                multi-class (optional) <- str: 
                    'multinomal' to select multi-class
                max_iter (optional) <- int: set the maximum number of iterations the solver takes
                penalty (optional) <- str: specify the penalty method for weights
                    'l1' for L1 norm
                    'l2' for L2 norm
                -------    
            
    -----------------------
    
    
    Decision Trees Regression
    -----------------------
        sklearn.linear_model.DecisionTreeRegressor
                desc: boolean decision tree for regression
                ------- 
                    max_depth (optional) <- int: max vertical depth of the tree
                ------- 
    -----------------------
    
    
    Decision Trees Classification
    -----------------------
        sklearn.linear_model.DecisionTreeClassifier
                desc: boolean decision tree for classification
                ------- 
                    max_depth (optional) <- int: max vertical depth of the tree
                ------- 
    -----------------------




sklearn.preprocessing.FunctionTransformer
sklearn.base.BaseEstimator
sklearn.base.TransformerMixin
sklearn.pipeline.Pipeline




"""
