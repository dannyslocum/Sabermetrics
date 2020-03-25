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


# steps
# 1. preprocessing
    # Fill in missing data
    # ----------------------
            # import sklearn.impute.MissingIndicator
            # import sklearn.impute.SimpleImputer
            # import sklearn.impute.KNNImputer
            # import sklearn.impute.IterativeImputer
            # pd.fillna()
    # ----------------------


    # Convert categorical to numerical
    # --------------------------------
            # import sklearn.preprocessing.OrdinalEncoder
            # import sklearn.preprocessing.OneHotEncoder
            # pd.get_dummies()
    # --------------------------------


    # Scale Data
    # -------------------------
            # import sklearn.preprocessing.scale
            # import sklearn.preprocessing.MinMaxScaler
            # import sklearn.preprocessing.StandardScaler
            # import sklearn.preprocessing.RobustScaler
            # import sklearn.preprocessing.MaxAbsScaler
            # import sklearn.preprocessing.Normalizer
    # ------------------------


    # Reduce Dimensionality
    # ------------------------
            # import sklearn.decomposition.PCA
            # import sklearn.cluster.FeatureAgglomeration
            """
                    desc: feature dimensionality reduction using agglomerative clustering 
                    ------- 
                        n_cluster <- int: choose number of clusters
                    ------- 
            """
            # LDA?
    # ------------------------

    
    # Other
    # -------------------------
            # import sklearn.preprocessing.LabelBinarizer
            # import sklearn.preprocessing.LabelEncoder
            # import sklearn.preprocessing.MultiLabelBinarizer

            # import sklearn.preprocessing.KBinsDiscretizer

            # import sklearn.preprocessing.Binarizer

            # import sklearn.preprocessing.PolynomialFeatures
    # -------------------------

    
# 2. split data
    # Train-Test Split
    # -------------------------
            # import sklearn.model_selection.train_test_split
            """
                    desc: least squares regression
                    ------- 
                        X <- dataset: add the dataset minus the labels
                        y <- list: add the labels only
                        test_size (optional) <- float: choose the percentage of test data to save
                    -------         
            """
    # -------------------------

    # Cross Validation
    # -------------------------
            # import sklearn.model_selection.cross_val_score
            """
            
                    desc: least squares regression
                    ------- 
                        estimator <- model(): choose a model to run the data against
                        X <- dataset: add the dataset minus the labels
                        y <- list: add the labels only
                        cv <- int: number of k folds
                        n_jobs <- int or None: choose number of CPUs (-1 means use all)
                    ------- 
            """
            # import sklearn.model_selection.GridSearchCV
            """
                    desc: least squares regression
                    ------- 
                        estimator <- model(): choose a model to run the data against
                        param_grid <- dict or list: names of parameters and values to test
                        scoring: metric to use for scoring
                        cv <- int: number of k folds
                        n_jobs <- int or None: choose number of CPUs (-1 means use all)
                    ------- 
                    ------- 
                        gcv = GridSearchCV(model, params, cv=4, iid=False)
                        gcv.fit(data, labels)
                        gcv.predict(new_data)
                    ------- 
                    ------- 
                        best_params_:
                        
                    ------- 
            """
    # -------------------------
                
                
# 3. choose and create model(s)
    # Linear regression
    # -----------------------
            # import sklearn.linear_model.LinearRegression
            """
                    desc: least squares regression
            """
            # import sklearn.linear_model.Ridge
            """
                    desc: least squares (L2 norm) but find smallest coefficients to be less susceptible to random noise
                    ------- 
                        alpha (optional) <- float: scaling factor for penalty term
                    ------- 
            """
            # import sklearn.linear_model.RidgeCV
            """
                    desc: Ridge with Cross Validation
                    ------- 
                        alphas <- List[float]: scaling factor for penalty term
                    ------- 
            """
            # import sklearn.linear_model.BayesianRidge
            """
                    desc: Ridge but with bayesian techniques by making assumptions about probability distribution prior to fitting model
                    ------- 
                        alpha_1 (optional) <- float: scaling factor for penalty term
                        alpha_2 (optional) <- float: scaling factor for penalty term
                        lambda_1 (optional) <- float: precision of the model's weights
                        lambda_2 (optional) <- float: precision of the model's weights
                    -------
            """
            # import sklearn.linear_model.Lasso
            """
                    desc: sparse linear regression using L1 norm). Typically reduced some features to weights of 0
                    ------- 
                        alpha (optional) <- float: scaling factor for penalty term
                    -------
            """
            # import sklearn.linear_model.LassoCV
            """
                    desc: Lasso with Cross Validation
                    ------- 
                        alphas <- List[float]: scaling factor for penalty term
                    --------
            """
    # -----------------------
    
    
    # Linear classification
    # -----------------------
            # import sklearn.linear_model.LogisticRegression
            """
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
            """
            # import sklearn.linear_model.LogisticRegressionCV
            """
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
            """
    # -----------------------
    

    # Clustering
    # -----------------------
            # import sklearn.metrics.cosine_similarity
            """
                    desc: comparison between data based on dot product and norm values
                    ------- 
                        x <- dataframe/list: comparison between rows or cells in the input
                        y (optional) <- dataframe/list: comparison between x and y
                    ------- 
            """
            # import sklearn.neighbors.NearestNeighbors
            """
                    desc: finds nearest neighbors
                    ------- 
                        k (optional) <- int: select number of nearest neighbors
                    ------- 
            """
            # import sklearn.cluster.KMeans
            """
                    desc: finds nearest neighbors, assumes clusters as spherical
                    ------- 
                        n_clusters <- int: select number of clusters
                    ------- 
            """
            # import sklearn.cluster.MiniBatchKMeans
            """
                    desc: finds nearest neighbors
                    ------- 
                        n_clusters <- int: select number of clusters
                        batch_size <- int: select size of each mini-batch
                    ------- 
            """
            # import sklearn.cluster.AgglomerativeClustering
            """
                    desc: hierarchical clusters, no assumptions prior to running about data
                          Agglomerative treats each object as a cluster, and merges until meeting the desired number, n
                    ------- 
                        n_clusters <- int: select number of clusters
                    ------- 
            """
            # import sklearn.cluster.MeanShift
            """
                    desc: assumes spherical clusters, finds n_clusters for you, high computation time
            """
            # import sklearn.cluster.DBSCAN
            """
                    desc: finds close and compact neighbors, finds n_clusters for you, faster
                    ------- 
                        eps <- float: measure for compactness
                        min_samples <- float: how many points needed to create a cluster
                    ------- 
            """


            # Cluster Evaluation
            # -----------------------
                    # import sklearn.metrics.adjusted_rand_score
                    """
                            desc: measurement of similarity betwee true labels and predicted labels
                                  Use if clusters are large and uniform in size
                    """
                    # import sklearn.metrics.adjusted_mutual_info_score
                    """
                            desc: measurement of similarity betwee true labels and predicted labels
                                  Use if clusters are unbalanced and small clusters exist
                    """
            # -----------------------
    # -----------------------


    # Decision Trees Regression
    # -----------------------
            # import sklearn.linear_model.DecisionTreeRegressor
            """
                    desc: boolean decision tree for regression
                    ------- 
                        max_depth (optional) <- int: max vertical depth of the tree
                    ------- 
            """
            # import sklearn.linear_model.RandomForestRegressor
            """
                    desc: 
                    ------- 

                    ------- 
            """
    # -----------------------
    
    
    # Decision Trees Classification
    # -----------------------
            # import sklearn.linear_model.DecisionTreeClassifier
            """
                    desc: boolean decision tree for classification
                    ------- 
                        max_depth (optional) <- int: max vertical depth of the tree
                    ------- 
            """
            # import sklearn.linear_model.RandomForestClassifier
            """
                    desc: 
                    ------- 

                    ------- 
            """
    # -----------------------


    # XGBoost
    # -----------------------
            # import xgboost.xgb
            """
                    desc: combine many decision trees efficiently
                    ------- 
                        max_depth (optional) <- int: max vertical depth of the tree
                        objective (optional) <- str: 
                        num_classes (optional) <- int: 
                    -------                     
                    ------- 
                        train = xgb.DMatrix(data, label=labels)
                        test = xgb.DMatrix(test_data, label=test_labels)
                        params = { max_depth: 0, objective: binary:logistic }
                        bst = xgb.train(params, train)
                            or for Cross Validation
                            cv_results = xgb.cv(params, dtrain, num_boost_round=10)
                        eval = bst.eval(test)
                        pred = bst.predict(<new_data>)
                        
                        bst.save_model('<name>.bin')
                        
                        bst = xgb.Booster()
                        bst.load_model('<name>.bin')
                    ------- 
            """
            # import xgboost.xgb.XGBClassifier
            """
                    desc: 
                    ------- 
                        objective <- str:
                    ------- 
                    ------- 
                        model = XGBClassifier()
                        model.fit(data)
                    ------- 
                    ------- 
                        model.feature_importance_:
                        xgb.plot_importance(model, importance_type='gain'):
                    ------- 
            """
            # import xgboost.xgb.XGBRegressor
            """
                    desc: 
                    ------- 
                        max_depth <- int:
                    ------- 
            """
    # -----------------------


    # Neural Network
    # -----------------------
            # import tensorflow as tf; import tensorflow-gpu as tf
            """
                    desc: 
                    ------- 

                    ------- 
                    ------- 
                            input_data = [[1.1, -0.3],[0.2, 0.1]]
                            input_data = [[1],[0]]

                            # num features input
                            input_size = 2
                            
                            # num features output
                            output_size = 2
                            
                            # initialize variables
                            input = tf.placeholder(tf.float32, shape=(None, input_size), name="input")
                            output = tf.placeholder(tf.int32, shape=(None, output_size), name="output")
                            
                            # create hidden layer with RELU activation
                            hidden1 = tf.layers.dense(input, activation=tf.nn.relu, name='hidden1')
                            hidden2 = tf.layers.dense(hidden1, activation=tf.nn.relu, name='hidden2')
                            # create fully connected layer logits, make into probabilities, get accuracy
                            logits = tf.layers.dense(hidden2, output_size, name='logits')
                            
                            # regression
                            probs = tf.nn.sigmoid(logits) 
                            rounded_probs = tf.round(probs)
                            predictions = tf.cast(rounded_probs, tf.int32)
                            is_correct = tf.equal(predictions, output)
                            is_correct_float = tf.cast(is_correct, tf.float32)
                            accuracy = tf.reduce_mean(is_correct_float)
                            
                            # classification
                            probs = tf.nn.softmax(logits) 
                            predictions = tf.argmax(probs, axis=-1)
                            class_labels = tf.argmax(outputs, axis=-1)
                            is_correct = tf.equal(predictions, class_labels)
                            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs,logits=logits)
                            
                            # get log-loss (cross entropy)
                            labels_float  = tf.cast(output, tf.float32)
                            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float, logits=logits)
                            loss = tf.reduce_mean(cross_entropy)
                            optimizer = tf.train.AdamOptimizer()
                            # optimizer = tf.train.GradientDescentOptimizer()
                            train_op = adam.minimize(loss)
                            
                            # train on actual data
                            init_op = tf.global_variables_initializer()
                            sess = tf.Session()
                            sess.run(init_op)
                            for i in range(1000):
                                feed_dict = {
                                    inputs: input_data[i],
                                    labels: input_labels[i]
                                }
                                sess.run(train_op, feed_dict=feed_dict)
                            
                            feed_dict = { inputs: test_data, labels: test_labels }
                            eval_acc = sess.run(accuracy, feed_dict=feed_dict)

                            


                    ------- 
                    ------- 

                    ------- 
            """
            # from keras.layers import Activation, Dense
            """
                    desc: 
                    ------- 
            
                    ------- 
                    ------- 
                        data = np.array([...])
                        labels = np.array([...])
                        
                        layer1 = Dense(5, input_dim=4) # input_dim is the feature dimensions
                        layer2 = Dense(3, activation='relu')
                        
                        # regression
                        layer3 = Dense(1, activation='relu')
                        
                        # classification
                        layer3 = Dense(3, activation='softmax')
                        
                        # model = Sequential()
                        # model.add(layer1)
                        # model.add(layer2)
                        model = Sequential([layer1, layer2, layer3])
                        
                        model.compile('adam', loss='binary_crossentropy' # 'categorical_crossentropy'
                                        metrics=['accuracy'])
                        train_output = model.fit(data, labels, batch_size=20, epochs=5)
                        
                        eval = model.evaluate(eval_data, eval_labels)
                        predictions = model.predict(new_data)
                    ------- 
                    ------- 
                        history: 
                        
                    -------                         
            """
# -----------------------


# 4. Choose Metric
    # Regression
    # ----------------------
            # import sklearn.metrics.r2_score
            # import sklearn.metrics.max_error
            # import sklearn.metrics.explained_variance_score
            # import sklearn.metrics.mean_squared_error
    # ----------------------
    
    
    # Classification
    # -----------------------
            # import sklearn.metrics.f1_score
            # import sklearn.metrics.accuracy_score
            # import sklearn.metrics.precision_score
            # import sklearn.metrics.recall_score
            # import sklearn.metrics.roc_auc_score
            # import sklearn.metrics.log_loss
            # import sklearn.metrics.jeccard_score
    # -----------------------


    # Clustering
    # ----------------------
            # import sklearn.metrics.adjusted_mutual_info_score
            # import sklearn.metrics.adjusted_rand_score
            # import sklearn.metrics.completeness_score
            # import sklearn.metrics.fowlkes_mallows_score
            # import sklearn.metrics.homogeneity_score
            # import sklearn.metrics.mutual_info_score
            # import sklearn.metrics.normalized_mutual_info_score
            # import sklearn.metrics.v_measure_score
    # ----------------------
    

# 5. Save model
    # Pickle
    # ----------------------
            # import pickle
            # s = pickle.dumps(clf)
            # clf2 = pickle.loads(s)
    # ----------------------

    # Joblib
    # ----------------------
            # from joblib import dump, load
            # dump(clf, 'filename.joblib')
            # clf2 = load('filename.joblib')
    # ----------------------

# import sklearn.preprocessing.FunctionTransformer
# import sklearn.base.BaseEstimator
# import sklearn.base.TransformerMixin
# import sklearn.pipeline.Pipeline
