import matplotlib.pyplot as pyt 
from pandas.plotting import scatter_matrix
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy import stats as stats
from sklearn.model_selection import *
from sklearn.model_selection import cross_validate
from sklearn import metrics   
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def my_LinReg(df,predictor,target,polynomial=None):
    
    train, test = train_test_split(df, test_size = 0.4)
    train = train.reset_index()
    test = test.reset_index()
    if polynomial is None:
        import statsmodels.api as sm
        X = sm.add_constant(train[predictor])
        Y=train[target]
        model = sm.OLS(Y,X)
        results = model.fit()
        print(results.summary())
        predict_train=results.predict(X)
        error= abs(Y-predict)
        print('\n Absolute mean error on training data',np.mean(error))
        error_test=abs(test[target]-results.predict(test[predictor]))
        print('\n Absolute mean error on test data',np.mean(error_test))
        return results
    else:
        # importing libraries for polynomial transform
        from sklearn.preprocessing import PolynomialFeatures
        # for creating pipeline
        from sklearn.pipeline import Pipeline
        # creating pipeline and fitting it on data
        Input=[('polynomial',PolynomialFeatures(degree=polynomial)),('modal',LinearRegression())]
        pipe=Pipeline(Input)
        model=pipe.fit(train[predictor],train[target])
        print('Model Summary',model.summary())
        poly_pred=pipe.predict(test[predictor])
        
        from sklearn.metrics import mean_squared_error
        print('\n\n============= RMSE for Polynomial Regression=>',np.sqrt(mean_squared_error(test[target],poly_pred)))
        return pipe



        

def my_LogRegClassifier(df1,list_predictors,string_target):
	train, test = train_test_split(df1, test_size = 0.4)
	train = train.reset_index()
	test = test.reset_index()
	features_train = train[list_predictors]
	label_train = train[string_target]
	features_test = test[list_predictors]
	label_test = test[string_target]
	
	clf = LogisticRegression()

	clf.fit(features_train,label_train)

	pred_train = clf.predict(features_train)
	pred_test = clf.predict(features_test)

	from sklearn.metrics import accuracy_score
	accuracy_train = accuracy_score(pred_train,label_train)
	accuracy_test = accuracy_score(pred_test,label_test)

	from sklearn import metrics
	fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
	auc_train = metrics.auc(fpr,tpr)

	fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
	auc_test = metrics.auc(fpr,tpr)

	print(accuracy_train,accuracy_test,auc_train,auc_test)
	pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])
	
	from ipywidgets import interact
	from bokeh.plotting import figure
	from bokeh.io import push_notebook, show, output_notebook
	output_notebook()

	
	preds = clf.predict_proba(features_test)[:,1]

	fpr, tpr, _ = metrics.roc_curve(np.array(label_test), preds)
	auc = metrics.auc(fpr,tpr)

	p = figure(title="ROC Curve - Test data")
	r = p.line(fpr,tpr,color='#0077bc',legend = 'AUC = '+ str(round(auc,3)), line_width=2)
	s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
	show(p)

	import matplotlib.pyplot as plt
	import scikitplot as skplt
	skplt.metrics.plot_cumulative_gain(label_test, preds)
	plt.show()
	return(clf)


def my_GbClassifier(df1,list_predictors,string_target):
	pd.options.mode.chained_assignment = None 
	train, test = train_test_split(df1, test_size = 0.4)
	train = train.reset_index()
	test = test.reset_index()
	features_train = train[list_predictors]
	label_train = train[string_target]
	features_test = test[list_predictors]
	label_test = test[string_target]
	n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]

	grid = {'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf}

	gb = GradientBoostingClassifier()

	gf_tune = GridSearchCV(estimator = gb, param_grid = grid, cv = 2, verbose=2, n_jobs = -1)
	gf_tune.fit(features_train, label_train)

	print(gf_tune.best_params_)


	clf = GradientBoostingClassifier(**gf_tune.best_params_)

	clf.fit(features_train,label_train)

	pred_train = clf.predict(features_train)
	pred_test = clf.predict(features_test)

	
	accuracy_train = accuracy_score(pred_train,label_train)
	accuracy_test = accuracy_score(pred_test,label_test)

	
	fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
	auc_train = metrics.auc(fpr,tpr)

	fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
	auc_test = metrics.auc(fpr,tpr)

	print(accuracy_train,accuracy_test,auc_train,auc_test)
	return(clf)



def my_RFClassifier(df1,list_predictors,string_target):

	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.ensemble import RandomForestClassifier
	train, test = train_test_split(df1, test_size = 0.4)
	train = train.reset_index()
	test = test.reset_index()
	features_train = train[list_predictors]
	label_train = train[string_target]
	features_test = test[list_predictors]
	label_test = test[string_target]

	n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
	max_features = ['auto', 'sqrt']
	max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
	max_depth.append(None)
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 4]
	bootstrap = [True, False]

	random_grid = {'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf,
	               'bootstrap': bootstrap}

	rf = RandomForestClassifier()

	rf_random = RandomizedSearchCV(estimator = rf, 
                                   param_distributions = random_grid,
                                   n_iter = 10, 
                                   cv = 2,
                                   verbose=2, 
                                   random_state=42,
                                   n_jobs = -1)
	rf_random.fit(features_train, label_train)

	print(rf_random.best_params_)

	clf = RandomForestClassifier(**rf_random.best_params_)

	clf.fit(features_train,label_train)

	pred_train = clf.predict(features_train)
	pred_test = clf.predict(features_test)

	accuracy_train = accuracy_score(pred_train,label_train)
	accuracy_test = accuracy_score(pred_test,label_test)

	
	fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
	auc_train = metrics.auc(fpr,tpr)

	fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
	auc_test = metrics.auc(fpr,tpr)

	print(accuracy_train,accuracy_test,auc_train,auc_test)
	pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])

	return(clf)



def my_LightGBMClassifierRegressor(df,predictors,target,ptype='binary'):
    
    print('\n Initiating Light gbm,make sure features are scaled')
    if ptype not in ['binary','multiclass','regression']: 
        print('\n Invalid value of type parameter passed.Exiting code')
        return
    
    #train_test_split 
    X_train,X_test,y_train,y_test=train_test_split(df[predictors],df[target],test_size=0.3,random_state=0)
    
    import lightgbm as lgb
    #Specifying the parameter
    params={}
    params['learning_rate']=0.03
    params['boosting_type']='gbdt' #GradientBoostingDecisionTree
    params['objective']=ptype #Binary target feature
    params['max_depth']=10
    if ptype in ['binary','multiclass']:
        params['metric']='binary_logloss' if ptype=='binary' else 'multi_logloss'
        if ptype =='multiclass': params['num_class']=df[target].nunique()
    
    #Converting the dataset in proper LGB format
    d_train=lgb.Dataset(X_train, label=y_train)
    #model creation and training
    clf=lgb.train(params,d_train,100)
    print(clf)
    
    #prediction on the test set
    y_pred_prob=clf.predict(X_test)
    
    
    if ptype=='binary':
        print()
        y_pred=y_pred_prob.apply(lambda x: 0 if x<= 0.5 else 1 )
        print('\n------------------------------------Binary classification performance-----------------------------------------\n')
        print(metrics.classification_report(y_test,y_pred))
        
        df_cfm=pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index=np.unique(y_test),columns=np.unique(y_test))
        print(df_cfm)
        
        

        fpr, tpr, _ = metrics.roc_curve(np.array(y_test), y_pred_prob)
        auc = metrics.auc(fpr,tpr)

        p = figure(title="ROC Curve - Test data")
        r = p.line(fpr,tpr,color='#0077bc',legend = 'AUC = '+ str(round(auc,3)), line_width=2)
        s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
        show(p)

        import matplotlib.pyplot as plt
        import scikitplot as skplt
        skplt.metrics.plot_cumulative_gain(y_test,y_pred_prob)
        plt.show()
        return(clf)
    
    if ptype=='multiclass':
        print()
        y_pred=[np.argmax(line) for line in y_pred_prob]
        
        print('\n------------------------------------Multicalss classification performance-----------------------------------------\n')
        print(metrics.classification_report(y_test,y_pred))
        print('\n Precison score:\n')
        print(precision_score(y_pred,y_test,average=None).mean())
        
        df_cfm=pd.DataFrame(metrics.confusion_matrix(y_test,y_pred), index=np.unique(y_test),columns=np.unique(y_test))
        print(df_cfm)
        return clf

   
    if ptype=='regression':
        
        print()
        predicted_y=round(y_pred_prob,3)
        
        print('\n------------------------------------Regression performance-----------------------------------------\n')
        print(metrics.r2_score(y_test, predicted_y))
        print(metrics.mean_squared_log_error(y_test, predicted_y))
        plt.figure(figsize=(10,10))
        sns.regplot(y_test, predicted_y, fit_reg=True, scatter_kws={"s": 100})
        return clf
              
   
              
        
              
              
              
        
        
              
    
