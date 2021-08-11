# Import Packages
import numpy as np
import pandas as pd

# Import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def make_algs(param_dicts = None):
    """
    Makes a list of tuples of classifier, regressor, and name for each algorithm. Currently KNN, DT, SVM, and RF.

    Returns:
         tups (list): List of tuples (classifier, regressor, name).
    """

    # Instantiate Classifiers
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    svc = SVC(probability=True)
    rf = RandomForestClassifier()
    clfs = [knn, dt, svc, rf]

    # Instantiate Regressors
    knnr = KNeighborsRegressor()
    dtr = DecisionTreeRegressor()
    svr = SVR()
    rfr = RandomForestRegressor()
    rgrs = [knnr, dtr, svr, rfr]

    # If param_dicts supplied, set parameters of algorithms
    if param_dicts:
        if 'KNN' in param_dicts.keys():
            knn_dict = param_dicts['KNN']
            knn.set_params(**knn_dict)
            knnr.set_params(**knn_dict)
        
        if 'DT' in param_dicts.keys():
            dt_dict = param_dicts['DT']
            dt.set_params(**dt_dict)
            dtr.set_params(**dt_dict)

        if 'SVM' in param_dicts.keys():
            svm_dict = param_dicts['SVM']
            svc.set_params(**svm_dict)
            svr.set_params(**svm_dict)

        if 'RF' in param_dicts.keys():
            rf_dict = param_dicts['RF']
            rf.set_params(**rf_dict)
            rfr.set_params(**rf_dict)

    # Provide names and zip together
    names = ['KNN', 'DT', 'SVM', 'RF']
    tups = list(zip(clfs, rgrs, names))

    return tups

def make_opt_algs(X, y, optimize_on = ['Continuous', 'Categorical']):
    """

    Optimizes algorithms via GridSearchCV and returns a tuple of optimized clfs and rgrs.

    Inputs:
    X (dataframe): Feature variables of a dataset.
    y (Series): Target variable of a dataset.
    optimize_on (String): 'Continuous' will optimize on continuous dataset, and 'Categorical' will optimize on categorical dataset.
    """

    # Instantiate Classifiers
    knn = KNeighborsClassifier(weights='distance')
    dt = DecisionTreeClassifier(max_features= 'auto', criterion= 'gini')
    svc = SVC(probability=True)
    rf = RandomForestClassifier(criterion= 'gini')
    clf_list = [knn, dt, svc, rf]

    # Instantiate Regressors
    knnr = KNeighborsRegressor(weights='distance')
    dtr = DecisionTreeRegressor(max_features= 'auto', criterion= 'mse')
    svr = SVR()
    rfr = RandomForestRegressor(criterion= 'mse')
    rgr_list = [knnr, dtr, svr, rfr]

    # Define rgr param grids
    dt_dict = {
        'max_depth': [50, 100, 200, 500],
        'min_samples_split': [2, 5, 10, 20, 40],
        'min_samples_leaf': [1, 5, 10, 20]
        }

    knn_dict = {
        'n_neighbors': np.arange(2, 22, 1)
        }

    svm_dict = {
        'kernel': ['linear', 'sigmoid', 'rbf'],
        'C': [1, 2, 5, 10]
        }

    rf_dict = {
        'n_estimators': [25, 50, 100, 200, 300, 500],
        'max_depth': [50, 100, 200, 500],
        'min_samples_split': [2, 5, 10, 20, 40],
        'min_samples_leaf': [1, 5, 10, 20]
        }

    if optimize_on == 'Continous':
        # Make rgr GridSearch
        knn_search = GridSearchCV(estimator = knnr, param_grid = knn_dict, n_jobs = -1, verbose= 10)
        dt_search = GridSearchCV(estimator = dtr, param_grid = dt_dict, n_jobs = -1, verbose= 10)
        svm_search = GridSearchCV(estimator = svr, param_grid = svm_dict, n_jobs = -1, verbose= 10)
        rf_search = GridSearchCV(estimator = rfr, param_grid = rf_dict, n_jobs = -1, verbose= 10)
        searches = [knn_search, dt_search, svm_search, rf_search]

    elif optimize_on == 'Categorical':
        # Make clf GridSearch
        knn_search = GridSearchCV(estimator = knn, param_grid = knn_dict, n_jobs = -1, verbose= 10)
        dt_search = GridSearchCV(estimator = dt, param_grid = dt_dict, n_jobs = -1, verbose= 10)
        svm_search = GridSearchCV(estimator = svc, param_grid = svm_dict, n_jobs = -1, verbose= 10)
        rf_search = GridSearchCV(estimator = rf, param_grid = rf_dict, n_jobs = -1, verbose= 10)
        searches = [knn_search, dt_search, svm_search, rf_search]

    # Make names and zip into tuples
    gridnames = ['KNNOpt', 'DTOpt', 'SVMOpt', 'RFOpt']

    # Fit searches
    for i,search in enumerate(searches):
        
        # Fit the gridsearch
        search.fit(X, y)
        opt_params = search.best_params_

        # Set the parameters of the regressor
        rgr = rgr_list[i]
        rgr.set_params(**opt_params)

        # Set the parameters of the classifier
        clf = clf_list[i]
        rgr.set_params(**opt_params)

    # Reinstantiate as new tuple and add to list
    names = ['KNN', 'DT', 'SVM', 'RF']
    tups = list(zip(clf_list, rgr_list, names))

    return tups

def makePipes(n_components = None, pred_prob = False):
    # Instantiate Preprocessing and PCA
    scale = StandardScaler()
    pca = PCA(n_components = n_components)

    # Instantiate Classifiers
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    if pred_prob == True:
        svc = SVC(probability= True)
    elif pred_prob == False:
        svc = SVC()
    rf = RandomForestClassifier()
    clfs = [knn, dt, svc, rf]

    # Instantiate Regressors
    knnr = KNeighborsRegressor()
    dtr = DecisionTreeRegressor()
    svr = SVR()
    rfr = RandomForestRegressor()
    rgrs = [knnr, dtr, svr, rfr]

    # Provide names and zip together
    names = ['KNN', 'DT', 'SVM', 'RF']
    clftups = list(zip(clfs, names))
    rgrtups = list(zip(rgrs,names))

    # Make pipes
    pipeclfs = [Pipeline(steps = [('Scale', scale), ('PCA', pca), ('KNN', knn)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('DT', dt)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('SVM', svc)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('RF', rf)])]
    pipergrs = [Pipeline(steps = [('Scale', scale), ('PCA', pca), ('KNN', knnr)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('DT', dtr)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('SVM', svr)]),
                Pipeline(steps=[('Scale', scale), ('PCA', pca), ('RF', rfr)])]
    pipenames = ['KNNPipe', 'DTPipe', 'SVMPipe', 'RFPipe']
    pipetups = list(zip(pipeclfs, pipergrs, pipenames))

    return pipetups

def makeOptPipes():
    """

    :return:
    """
    # Instantiate Preprocessing and PCA
    scale = StandardScaler()
    pca = PCA()

    # Instantiate Classifiers
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(max_features= 'auto', criterion= 'gini')
    svc = SVC()
    rf = RandomForestClassifier(criterion= 'gini')

    # Instantiate Regressors
    knnr = KNeighborsRegressor()
    dtr = DecisionTreeRegressor(max_features= 'auto', criterion= 'mse')
    svr = SVR()
    rfr = RandomForestRegressor(criterion= 'mse')

    # Define rgr param grids
    dt_dict = {'PCA__n_components': np.arange(10, 160, 10),
               'DT__max_depth': [50, 100, 200, 500],
               'DT__min_samples_split': [2, 5, 10, 20, 40],
               'DT__min_samples_leaf': [1, 5, 10, 20]
               }
    knn_dict = {'PCA__n_components': np.arange(10, 160, 10),
                'KNN__n_neighbors': np.arange(2, 22, 2)
                }
    svm_dict = {'PCA__n_components': np.arange(10, 160, 10),
                'SVM__kernel': ['linear', 'sigmoid', 'rbf'],
                'SVM__C': [1, 2, 5, 10]
                }
    rf_dict = {'PCA__n_components': np.arange(10, 160, 10),
               'RF__n_estimators': [100, 200, 300, 500],
               'RF__max_depth': [50, 100, 200, 500],
               'RF__min_samples_split': [2, 5, 10, 20, 40],
               'RF__min_samples_leaf': [1, 5, 10, 20]
               }

    # Make clf pipes
    knn_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('KNN', knn)])
    dt_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('DT', dt)])
    svm_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('SVM', svc)])
    rf_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('RF', rf)])
    clf_pipes = [knn_pipe, dt_pipe, svm_pipe, rf_pipe]

    # Make rgr pipes
    knnr_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('KNN', knnr)])
    dtr_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('DT', dtr)])
    svmr_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('SVM', svr)])
    rfr_pipe = Pipeline(steps = [('Scale', scale), ('PCA', pca), ('RF', rfr)])
    rgr_pipes = [knnr_pipe, dtr_pipe, svmr_pipe, rfr_pipe]

    # Make rgr GridSearch
    knn_search = GridSearchCV(estimator = knnr_pipe, param_grid = knn_dict, n_jobs = -1)
    dt_search = RandomizedSearchCV(estimator = dtr_pipe, param_distributions = dt_dict, n_jobs = -1)
    svm_search = GridSearchCV(estimator = svmr_pipe, param_grid = svm_dict, n_jobs = -1)
    rf_search = RandomizedSearchCV(estimator = rfr_pipe, param_distributions = rf_dict, n_jobs = -1)
    rgr_searches = [knn_search, dt_search, svm_search, rf_search]

    # Make names and zip into tuples
    gridnames = ['KNNOpt', 'DTOpt', 'SVMOpt', 'RFOpt']
    gridtups = list(zip(rgr_searches, rgr_pipes, clf_pipes, gridnames))

    return gridtups

def modelOpt(X = None, y = None, tups = None):
    """

    Parameters:
        X
        y
        tups

    Returns:
        opt_tups
    """
    opt_tups = []
    for tuple in tups:

        # Get name
        name = tuple[3]

        # Get best params for rgr and give to rgr pipe
        rgr_search = tuple[0]
        rgr_search.fit(X, y)
        rgr_opt_params = rgr_search.best_params_
        rgr_pipe = tuple[1]
        rgr_pipe.set_params(**rgr_opt_params)

        # Give rgr opt params to clf pipe
        clf_pipe = tuple[2]
        clf_pipe.set_params(**rgr_opt_params)

        # Reinstantiate as new tuple and add to list
        opt_tup = (clf_pipe, rgr_pipe, name)
        opt_tups.append(opt_tup)

    return opt_tups
