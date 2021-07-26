#check unique + null
'''
# for col in df_32.columns:
#     if df_32[col].isna().values.sum() != 0:
#         print(f'{col}:{df_32[col].isna().values.sum()}, ({df_32[col].dtypes})')
#         print(f'{col}:', df_32[col].unique())
'''
def check_df(df, show_all=False, show_unique=False):
    null_exist =False
    for col in df.columns:
        if show_all or df[col].isna().values.sum() != 0:
            print(f'{col}: null={df[col].isna().values.sum()}, dtypes={df[col].dtypes}')
            if show_unique:
                print(f'{col}:', df[col].unique())
            if df[col].isna().values.sum() != 0:
                null_exist =True
    
    if not null_exist:
        print('No null exist in this dataframe.')

#cross validate models to know the model performance on particular columns 
#to fill in the missing data evaluation
def cross_validate_models(models, train_data, col):
    '''
    https://stats.stackexchange.com/questions/184860/repeated-k-fold-cross-validation-vs-repeated-holdout-cross-validation-which-ap
    https://oralytics.com/2020/09/21/k-fold-repeated-k-fold-cross-validation-in-python/
    https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
    https://www.researchgate.net/post/Repeated-N-fold-cross-validation
    https://datascience.stackexchange.com/questions/28158/how-to-calculate-the-fold-number-k-fold-in-cross-validation
    guide on R2 vs RMSE - https://stats.stackexchange.com/questions/270746/comparing-2-models-with-very-different-r2-values-but-with-very-close-rmse-values
    guide on std - https://www.researchgate.net/post/What-do-you-consider-a-good-standard-deviation
    Coefficient of Variable=standard deviation / mean).
    CV >= 1 indicates a relatively high variation, while a CV < 1 can be considered low.
    '''
    features = ['Models','test_r2_mean', 'test_r2_std', 'train_r2_mean', 'train_r2_std',
                'test_neg_rmse_mean', 'test_neg_rmse_std', 'coef_var_rmse',
                'train_neg_rmse_mean', 'train_neg_rmse_std',
                'test_explained_variance_mean', 'test_explained_variance_std', 
                'y_mean']
    scoring =['r2', 'neg_root_mean_squared_error', 'explained_variance']
    
    df = pd.DataFrame(columns = features) 

    best_rmse_mean = -9999999
    best_model = None
    for index, model in enumerate(models):
        #drop all columns that has NAN
#             X = df_32_X.drop(col, axis=1)
        X = train_data.drop(col, axis=1)
        y = train_data[col]
        rkf = RepeatedKFold(n_splits=5, n_repeats=15, random_state=42) #since dataset is low, cv=~5 (5 folds) and repeat 15 times
        cv = cross_validate(model, X, y, cv=rkf, return_train_score=True, scoring=scoring)

        info = pd.Series({'Models':model, 
                          'test_r2_mean':cv['test_r2'].mean(), 'test_r2_std':cv['test_r2'].std(), 
                          'train_r2_mean': cv['train_r2'].mean(), 'train_r2_std': cv['train_r2'].std(), 
                          'test_neg_rmse_mean': cv['test_neg_root_mean_squared_error'].mean(), 
                          'test_neg_rmse_std': cv['test_neg_root_mean_squared_error'].std(),
                          'coef_var_rmse': -(cv['test_neg_root_mean_squared_error'].std() / cv['test_neg_root_mean_squared_error'].mean()), 
                          'train_neg_rmse_mean': cv['train_neg_root_mean_squared_error'].mean(),
                          'train_neg_rmse_std': cv['train_neg_root_mean_squared_error'].std(),
                          'test_explained_variance_mean': cv['test_explained_variance'].mean(),
                          'test_explained_variance_std': cv['test_explained_variance'].std(),
                          'y_mean': y.mean()
                         })
#         print('here:')
#         print(cv['test_r2'])
                     
        df.loc[index] = info

        if best_rmse_mean < cv['test_neg_root_mean_squared_error'].mean():
            best_rmse_mean = cv['test_neg_root_mean_squared_error'].mean()
            best_model = model
    
    print(f'Cross validation Result for {col}:')
    display(df)
    print(f'The best model is \n {best_model} \n with rmse score: {best_rmse_mean}')
    print(error_gt_pred(best_rmse_mean, y.mean()))
    return best_model, best_rmse_mean

# function for KNN model-based imputation of missing values using features without NaN as predictors
def impute_knn(df):
    cols_nan = df.columns[df.isna().any()].tolist()
    cols_no_nan = df.columns.difference(cols_nan).values
    for col in cols_nan:
        test_data = df[df[col].isna()]
        display(test_data)
        train_data = df.dropna()
        knr = KNeighborsRegressor(n_neighbors=5).fit(train_data[cols_no_nan], train_data[col])
        rows_imputed = df.loc[df[col].isna(), col]
        df.loc[df[col].isna(), col] = knr.predict(test_data[cols_no_nan])
        print('Cols imputed', col)
        print('Rows imputed:', len(rows_imputed))
    return df

#features
def test_features(models, X, label_name): #backward
    test_features = [i for i in X.columns if i !=label_name]
    df_list = []
    prev_good_features = []
    bad_features = ['temp']
    loop_count = 1
    while bad_features!=[]:
        print(f'Loop {loop_count}:')
        bad_features = []
        good_features = []
        index = 0
        features = ['features', 'good', 'remarks']
        df = pd.DataFrame(columns = features) 
        
        X_to_test = X[test_features]
        X_to_test[label_name] = X[label_name]
        _, base_rmse = cross_validate_models(models, X_to_test, label_name)
        print(f'features to test: {test_features}')

        for f in tqdm(test_features):
            _, test_rmse = cross_validate_models(models, X_to_test.drop(f, axis=1), label_name)
            if test_rmse > base_rmse: #after dropping, if the model gets better, then the feature is bad:
                bad_features.append(f)
                info = pd.Series({'features':f, 'good':0, 'remarks': f'{test_rmse - base_rmse}'})        
            else:
                good_features.append(f)
                info = pd.Series({'features':f, 'good':1, 'remarks': f'{test_rmse - base_rmse}'}) 

            df.loc[index] = info
            index += 1
        
        test_features = good_features.copy()
        df_list.append(df)
        
        if prev_good_features == good_features:
            break;
        else:
            prev_good_features = good_features.copy()
            
        loop_count +=1
        
    return df_list

def test_features_forward(models, X, label_name, col_approved): #forward
    
    df_list = []
    prev_good_features = []
    bad_features = ['temp']
    loop_count = 1
    
    while bad_features!=[]:
        
        test_features_master = [i for i in X.columns if i !=label_name and i not in col_approved]
        print(f'Loop {loop_count}:')
        bad_features = []
        good_features = []
        index = 0
        features = ['features', 'good', 'remarks']
        df = pd.DataFrame(columns = features) 
        
        X_to_test = X[col_approved]
        X_to_test[label_name] = X[label_name]
        _, base_rmse = cross_validate_models(models, X_to_test, label_name)
        best_feature_rmse = base_rmse.copy()
        best_feature = None
        
        print(f'features to test: {test_features_master}')

        for f in tqdm(test_features_master):
            _, test_rmse = cross_validate_models(models, X_to_test.join(X[f]), label_name)
            if test_rmse > base_rmse: #after putting in, if the model gets better, then the feature is good:
                bad_features.append(f)
                info = pd.Series({'features':f, 'good':1, 'remarks': f'{test_rmse - base_rmse}'})        
            else:
                good_features.append(f)
                info = pd.Series({'features':f, 'good':0, 'remarks': f'{test_rmse - base_rmse}'}) 
                
            if test_rmse > best_feature_rmse:
                best_feature_rmse = test_rmse
                best_feature = f
                best_feature_to_return = X_to_test.columns

            df.loc[index] = info
            index += 1
        
        df_list.append(df)
        
        print('--------------------')
        print(f'In Loop {loop_count}:')
        print(f'the best rmse: {best_feature_rmse}')
        print(f'feature chosen to be added: {best_feature}')
        print(f'col_approved now: {col_approved}')
        print('--------------------')
        
        if best_feature_rmse == base_rmse:
            print('rmse same. Stop!')
            break
        
        if best_feature != None:
            col_approved.append(best_feature)
        else:
            break
            
        loop_count +=1
        
    return df_list, best_feature_rmse, best_feature_to_return
            

# def important_features(X, y, best_model, estimators=1000):
#     '''
#     https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/
#     The higher the score, the better
    
#     '''
#     # Building the model
#     best_model = RandomForestClassifier(n_estimators = estimators, criterion ='entropy', max_features = 2)
    
#     # Training the model
# #     forest.fit(X, y)
#     best_model.fit(X, y)

#     # Computing the importance of each feature
#     feature_importance = best_model.feature_importances_

#     # Normalizing the individual importances
#     tfi = [tree.feature_importances_ for tree in best_model.estimators_]
#     feature_importance_normalized = np.std(tfi, axis = 0)

#     #axis =0 --> iterate through rows and return 1 val
#     #axis =1 --> iterate through columns and return 1 val

#     # Plotting a Bar Graph to compare the models
    
#     try:
#         feature_importance_best_model = best_model.coef_
#     except:
#         feature_importance_best_model = best_model.feature_importances_
    
#     df = pd.DataFrame({"X_col":X.columns, "Feature Importances":feature_importance_best_model})
#     df = df.sort_values(by=['Feature Importances'], ascending=False)
    
#     x_col = list(df['X_col'])
#     y_f = list(df['Feature Importances'])
    
#     plt.bar(x_col, y_f)
#     plt.xlabel(f'Feature Labels on {y.name}')
#     plt.ylabel('Feature Importances')
#     plt.title('Comparison of different Feature Importances')
#     plt.xticks(range(len(x_col)), x_col, rotation=90)
#     plt.show()
# #     plt.bar(X.columns, feature_importance_normalized)
# #     plt.xlabel(f'Feature Labels on {y.name}')
# #     plt.ylabel('Feature Importances')
# #     plt.title('Comparison of different Feature Importances')
# #     plt.xticks(range(len(X.columns)), X.columns, rotation=90)
# #     plt.show()

def error_gt_pred(rmse, y_mean):
    error = 100 * round(abs(rmse/y_mean),3)
    print(f'The error:{error}%')
    return error
    
def backward_elimination(x,y,SL, const=False):
    '''
    https://www.javatpoint.com/backward-elimination-in-machine-learning
    https://www.kaggle.com/saikrishna20/1-3-multiple-linear-regression-backward-eliminat
    1. get rid of the column with highest pvalues.
    2. compare the previous model vs after removed model.
    3. if previous adjusted r2 > after adjusted r2, stop the model, rollback
    4. else, repeat step 1.
    '''
    if const == True:
        x['const'] = 1
    
    numVars = len(x.columns)
    columns_drop_list = []
    for i in range(0, numVars):
        regressor_OLS = smrlm.OLS(endog = y, exog = x).fit()
        maxVar = max(regressor_OLS.pvalues)
        adjR_before = regressor_OLS.rsquared_adj
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    temp = x.iloc[:, j]
                    columns_drop_list.append(x.columns[j])
                    x = x.drop(x.columns[j], axis=1)
                    tmp_regressor = smrlm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj
                    if (adjR_before >= adjR_after):
                        x = x.join(temp)
                        print (regressor_OLS.summary())
                        return x, columns_drop_list
                    else:
                        continue
    regressor_OLS.summary()
    return x, columns_drop_list

def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    '''
    https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/
    go forwards first, then stop, then use the 'best features' to fit again (backward), then select out the best features.
    '''
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                print(best_features)
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features

def plot_heat_map(X):
    corr = X.corr()
    top_features = corr.index
    plt.figure(figsize=(20,20))
    sns.heatmap(X[top_features].corr(),annot=True)
    
def features_graph(X):
    X_features = X.drop(label_name,axis=1)
    y = X[label_name]

    columns_list = []
    for i in range(0, len(X_features.columns), 3):
        columns_list.append(X_features.columns[i:i+3])

    for col in columns_list:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))
        ax1.scatter(X_features[col[0]], y)
        ax1.set_title(f'{col[0]} vs {y.name}')
        ax1.set_xlabel(col[0])
        ax1.set_ylabel(y.name)

        ax2.scatter(X_features[col[1]], y)
        ax2.set_title(f'{col[1]} vs {y.name}')
        ax2.set_xlabel(col[1])
        ax2.set_ylabel(y.name)

        ax3.scatter(X_features[col[2]], y)
        ax3.set_title(f'{col[2]} vs {y.name}')
        ax3.set_xlabel(col[2])
        ax3.set_ylabel(y.name)
        
def hadlock_1(ac_cm, fl_cm):
    x = 1.304+0.05281*ac_cm+0.1938*fl_cm -0.004*ac_cm*fl_cm
    return 10**x

#tuning

#--------------------------------------linear_regression ----------------------------------------
def linear_regression_hypertune(X, y, cv):
    linear_regression_model = LinearRegression()
    # define search space
    linear_regression_space = dict()
    linear_regression_space['fit_intercept'] = [True, False]
    linear_regression_space['normalize'] = [True, False]
    # define search
    search = RandomizedSearchCV(linear_regression_model, linear_regression_space, 
                                n_iter=4, scoring='neg_mean_absolute_error',  #since only has 2^2, so 4 is the max
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score (neg_mean_absolute_error): %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------------------------linear_regression --------------------

#---------------------------------------------------- Ridge ------------------------------------------------------------
def ridge_hypertune(X, y, cv):
    ridge_model = Ridge()
    # define search space
    ridge_space = dict()
    #solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}, default=’auto’
    ridge_space['solver'] = ['auto','svd', 'cholesky', 'lsqr','sparse_cg', 'sag', 'saga']
    ridge_space['alpha'] = loguniform(1e-5, 100)
    ridge_space['fit_intercept'] = [True, False]
    ridge_space['normalize'] = [True, False]
    # define search
    search_ridge = RandomizedSearchCV(ridge_model, ridge_space, n_iter=1000, 
                                scoring='neg_root_mean_squared_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search_ridge.fit(X, y)
    # summarize result
    print('Best Score in neg_root_mean_squared_error: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#----------------------------------------------------Ridge ------------------------------------------------------------

#------------------------------------------Bayesian Ridge Regressor ------------------------------------------------------------
def b_ridge_hypertune(X, y, cv):
    b_ridge_model = linear_model.BayesianRidge(n_iter=500)
    # define search space
    b_ridge_space = dict()
    b_ridge_space['alpha_1'] = loguniform(1e-6, 10)
    b_ridge_space['alpha_2'] = loguniform(1e-6, 10)
    b_ridge_space['lambda_1'] = loguniform(1e-6, 10)
    b_ridge_space['lambda_2'] = loguniform(1e-6, 10)
    b_ridge_space['fit_intercept'] = [True, False]
    b_ridge_space['normalize'] = [True, False]
    
    # define search
    search_b_ridge = RandomizedSearchCV(b_ridge_model, b_ridge_space, n_iter=1000, 
                                scoring='neg_root_mean_squared_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search_b_ridge.fit(X, y)
    # summarize result
    print('Best Score in neg_root_mean_squared_error: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------Bayesian Ridge Regressor ------------------------------------------------------------


#----------------------------------------------------lasso ------------------------------------------------------------
def lasso_hypertune(X, y, cv):
    lasso_model = Lasso()
    # define search space
    lasso_space = dict()
    lasso_space['alpha'] = loguniform(1e-5, 100)#randomly choose value within this range
    lasso_space['selection'] = ['cyclic', 'random']
    lasso_space['fit_intercept'] = [True, False]
    lasso_space['normalize'] = [True, False]
    # define search
    search = RandomizedSearchCV(lasso_model, lasso_space, 
                                n_iter=1000, scoring='neg_mean_absolute_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#----------------------------------------------------lasso ------------------------------------------------------------

#------------------------------------------RandomForestRegressor------------------------------------------------
def rf_hypertune(X, y, cv):
    random_forest_model = RandomForestRegressor()
    # define search space
    random_forest_space = dict()
    random_forest_space['n_estimators'] = randint(2, 100)
    random_forest_space['max_depth'] = randint(1, 100) 
    random_forest_space['min_samples_split'] = randint(2, 100)#must greater than 1
    random_forest_space['min_samples_leaf'] = randint(1, 100)
    random_forest_space['max_features'] = ['auto', 'sqrt', 'log2']
    random_forest_space['max_leaf_nodes'] = randint(1, 100)

    # define search
    search = RandomizedSearchCV(random_forest_model, random_forest_space, 
                                n_iter=1000, scoring='neg_mean_absolute_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------RandomForestRegressor------------------------------------------------

#------------------------------------------------------------svr ------------------------------------------------------------
#https://stackoverflow.com/questions/51459406/how-to-apply-standardscaler-in-pipeline-in-scikit-learn-sklearn
#svr will error if use RandomizedSearchCV, so i tried Grid Search instead.
def svr_hypertune(X, y, cv):
    pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVR())]) #SVR = regression, SVC = classification

    param_grid = dict(clf__C=[0.1, 1, 10, 100, 1000],
                      clf__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                      clf__epsilon= [0.01, 0.1, 1, 10, 100, 1000],
                      clf__gamma= ['scale', 'auto'],
                     )

    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=4, verbose=10, scoring= 'neg_mean_absolute_error')
    result = search.fit(X, y)

    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------------------------svr --------------------------------------------

#------------------------------------------AdaBoostRegressor------------------------------------------------
def ada_hypertune(X, y, cv):
    ada_model = AdaBoostRegressor()
    # define search space
    ada_space = dict()
    ada_space['loss'] = ['linear', 'square', 'exponential']
    ada_space['learning_rate'] = loguniform(1e-5, 100) #range from 0.00001 to 100 
    ada_space['n_estimators'] = randint(2, 100)

    # define search
    search = RandomizedSearchCV(ada_model, ada_space, 
                                n_iter=1000, scoring='neg_mean_absolute_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------AdaBoostRegressor------------------------------------------------

#------------------------------------------ GradientBoostingRegressor------------------------------------------------
def gbr_hypertune(X, y, cv):
    gbr_model = GradientBoostingRegressor(loss='ls', n_estimators=500, n_iter_no_change=50)
    # define search space
    gbr_space = dict()
    gbr_space['loss'] = ['ls', 'lad', 'huber', 'quantile']
    gbr_space['alpha'] = loguniform(1e-5, 1) #range from 0.00001 to 1
    gbr_space['learning_rate'] = loguniform(1e-5, 100) #range from 0.00001 to 100 
    gbr_space['n_estimators'] = randint(2, 100)
    gbr_space['subsample'] = loguniform(1e-5, 1)
    gbr_space['min_samples_split'] = randint(2, 100)#must greater than 1
    gbr_space['min_samples_leaf'] = randint(1, 100)
    gbr_space['max_depth'] = randint(1, 20)
    gbr_space['min_impurity_decrease'] = loguniform(1e-5, 1)
    gbr_space['max_features'] = ['auto', 'sqrt', 'log2']
    gbr_space['max_leaf_nodes'] = randint(1, 100)
    gbr_space['ccp_alpha'] = loguniform(1e-5, 1)
    
    # define search
    search = RandomizedSearchCV(gbr_model, gbr_space, 
                                n_iter=1000, scoring='neg_mean_absolute_error', 
                                n_jobs=-1, cv=cv, random_state=1, verbose=10)
    # execute search
    result = search.fit(X, y)
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    return result.best_params_
#------------------------------------------ GradientBoostingRegressor------------------------------------------------
