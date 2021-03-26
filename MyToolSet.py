
import matplotlib.pyplot as pyt 
from pandas.plotting import scatter_matrix
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import itertools
import numpy as np
# from fancyimpute import MICE as MICE
from scipy import stats as stats
from scipy.special import boxcox1p
from imblearn.combine import SMOTETomek
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

# Define function for scatter plot
def my_plot(data,target=None):
    
    hist_by_category(data,target)
    sns.set(style="ticks", color_codes=True)
     
    
    if target==None:
        data['idx']='All'
        target='idx'
    sns.pairplot(data, hue=target, palette="husl")
    plt.show()
    if target==None:
        data=data.drop(['idx'])
    return data
###################################################################################################

def pre_process(data,scale=None):
    # Data transformation
    # Convert categorical values to numeric using label encoder
    df=data.copy()
    from sklearn import preprocessing
    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)
    if scale in ['minmax','standard']:
        df=MyFeatureScale(df,scale)
        print('\n Scaling of features completed')

    # Encoding the categorical variable
    print('\n Encoding categories')
    print(df.select_dtypes(exclude=np.number).columns)
    fit = df.select_dtypes(exclude=np.number).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))
    #Convert the categorical columns based on encoding
    for i in list(d.keys()):
        df[i] = d[i].transform(df[i].fillna('NA'))
    return(df)

# Define function for outlier\boxplot

def MyFeatureScale(df,scale):
    
    if scale=='minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        # fit scaler on training data
    if scale=='standard':
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()        
        # transform the training data column
    
    df_new=pd.DataFrame(scaler.fit_transform(df.select_dtypes(np.number)), columns=df.select_dtypes(np.number).columns)
    return pd.concat([df_new,df.select_dtypes(exclude=np.number)],axis=1)
##############################################################################################################################
def outlier_flagbyZscore(data):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=np.number)
    print('Original data:'+str(data.shape))
    z = np.abs(stats.zscore(data))
    data=data[(z < 3).all(axis=1)]
    print('\nRemoval of outliers'+str(data.shape))
    return(data)
    
def box_plot(data):
    print(data.describe())
    sns.boxplot(data.select_dtypes(np.number), orient="h",linewidth=2.5, palette="Set2")
    plt.show()


# Define function for bar plot by category

def hist_by_category(data,target):

    for col in data.select_dtypes(exclude=np.number):
            sns.countplot(data[col])
            plt.show()
     
    return


#Define function to impute missing data

def impute_values(df):
    #fancy impute removes column names.
    train_cols = df.columns
    # Use MICE to fill in each row's missing features
    df = pd.DataFrame(MICE(verbose=False).complete(df))
    df.columns = train_cols
    return(df)
#####################################################################################################
def impute_all(data,ignore=[]):
    df=pd.DataFrame()
    
    data = data.replace(r'^\s+$', np.nan, regex=True)
    categorical_columns = data.select_dtypes(exclude=np.number).columns.difference(ignore)
    numeric_columns = data.select_dtypes(include=np.number).columns.difference(ignore)
    print('Numerics',numeric_columns)
    print('Categories',categorical_columns)
    for ncol in numeric_columns:
        df[ncol]=data[ncol].astype(float)
    for catcol in categorical_columns:
        df[catcol]=data[catcol].astype(str)
    print('\n')
    mis_val_table_ren_columns=missing_values_table(df)
    if mis_val_table_ren_columns.shape[0]==0 :
        print('No missing values found in given data')
        return data
    for col in df.columns.difference(ignore):
        print('Treating--> '+col)
        if col not in mis_val_table_ren_columns['Missing Values']:
            print(col+' Has no missing value.Skipping to next one')
            continue
        attributes=list(df.columns.difference(ignore))
        attributes.remove(col)
        
        df[col]=knn_impute(df[col],
                             df[attributes],
                             aggregation_method="mode" if col in categorical_columns else 'median', 
                             k_neighbors=4, 
                             numeric_distance='euclidean',
                             categorical_distance='hamming')
    
    
    
    print(missing_values_table(df))
    print("\n --------------------------- The main imputation has been completed. Checking corner cases.\n\n")
    from sklearn.impute import SimpleImputer
    
    
    imp=SimpleImputer(strategy="most_frequent")
    
    for col in df.columns.difference(ignore):
        
        df[col]=imp.fit_transform(df[[col]]).ravel()
          
       
    print('\n Imputation complete.Take a look at the final set.')
    print(missing_values_table(df))
    for catcol in categorical_columns:
        df[catcol]=df[catcol].astype('object')
    df.set_index(data.index,inplace=True)
    return(df)
######################################################################################################################
def outlier_detection(df):
    #specify  column names to be modelled
    print('\n --- Outlier removal process triggered,with shape of data,',df.shape)   
    to_model_columns=df.select_dtypes([np.number]).columns
    from sklearn.ensemble import IsolationForest
    
    clf=IsolationForest(n_estimators=25,
                        max_samples='auto',
                        
                        contamination=float(.05),
                        max_features=1.0,
                        bootstrap=False,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0)
    
    clf.fit(df[to_model_columns])
    pred = clf.predict(df[to_model_columns])
    df['anomaly']=pred
    outliers=df.loc[df['anomaly']==-1]
    outlier_index=list(outliers.index)
    pcnt=100* len(outlier_index)/len(df)
    print("\n----------------------------------- Percentage of data points flagged as outliers ",str(round(pcnt,2))," %---------------")
    #Find the number of anomalies and normal points here points classified -1 are anomalous
    print(df['anomaly'].value_counts())
    
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(df[to_model_columns])
    res=pd.DataFrame(pca.transform(df[to_model_columns]))
    Z = np.array(res)
    plt.title("IsolationForest")
    plt.contourf( Z, cmap=plt.cm.Blues_r)
     
    b1 = plt.scatter(res[0], res[1], c='green',
                     s=20,label="normal points")
     
    b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
    plt.legend(loc="upper right")
    plt.show()
    
    return(df,df[df.anomaly==1][df.columns.difference(['anomaly'])])
#########################################################################################################################
def my_featureTransformToNormalDist(column,name,lamb=None):
    print('\nInitiating box cox transformation')
    list_lambda=list()
    
    original=col
    transformed_col, best_lambda = boxcox(df[col],lamb)
    sns.kdeplot(pd.DataFrame([original,transformed_col],columns=['Original','Transformed']),x="Box Cox transformation of "+str(name))
    print("\n Check for normality from the plot and store lambda value to apply to test data later")
    return transformed_col, best_lambda



def missing_values_table(df):
        print(df.dtypes)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns



def BalancedSample(df,target):
    from imblearn.combine import SMOTEENN
    
    columns=df.columns.difference([target])
    print('the data originally has a shape, ',df[target].value_counts())
    X_smt, y_smt = SMOTEENN().fit_sample(df[columns],df[target])
    X_smt=pd.DataFrame(X_smt, columns=columns)
    X_smt[target]=y_smt
    print('the data now has a shape, ',X_smt[target].value_counts())
    

    return(X_smt)

def My_PrincipalComponentAnalysis(df,num=None):
    print('\n Principal Component Analysis Triggered')
    df=df.select_dtypes(include=np.number)
    mean_values= np.round(df.mean(axis=0))
    sd_values=np.round(df.std(axis=0))
    if num==None: num=df.shape[1]
    flag=False if (mean_values.max()==0) & (sd_values.min()==1) else True
    if flag:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print('Data is not scaled. Applying Standard Scaling with mean 0 ans s.d=1')
        df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    from sklearn.decomposition import PCA
    pca=PCA(num)
    principalComponents = pca.fit_transform(np.nan_to_num(df))
    
    print(np.round(pca.explained_variance_ratio_, 3))
    sing_vals = np.arange(num) + 1

    fig = plt.figure(figsize=(8,5))
    plt.plot(sing_vals[:12], np.round(pca.explained_variance_ratio_, 3)[:12], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    leg = plt.legend(['Variance explained'], loc='best', borderpad=0.3, 
                     shadow=False, 
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    plt.show()
    factor_list=['Component-'+str(idx+1) for idx in range(num)]
    pca_df=pd.DataFrame(pca.components_.T, columns= factor_list,index=df.columns)
    maxLoading = pd.DataFrame(principalComponents,columns=factor_list)
       
    return pca_df,maxLoading

from factor_analyzer.factor_analyzer import calculate_kmo

from factor_analyzer.factor_analyzer import calculate_kmo

def My_FactorAnalyzer(df,num=None):
    print('\n Factor Analysis has been triggered')
    mean_values= np.round(df.mean(axis=0))
    sd_values=np.round(df.std(axis=0))
    if num==None: num=df.shape[1]
    flag=False if (mean_values.max()==0) & (sd_values.min()==1) else True
    if flag:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print('Data is not scaled. Applying Standard Scaling with mean 0 ans s.d=1')
        df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print('Determining suitability of the Factor Analysis')
    _,kmo_model = calculate_kmo(df)
    print(kmo_model)
    print('\n Look for the optimal number of factors. Eigen Values below 1 are the ones which can be ignored in the scree plot')
    print('\n Eigen Values essentially explain Variance and greater than 1 implies more variance than the standardised individual feature.')
    
    from factor_analyzer import FactorAnalyzer
    
    fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=num)
    fa.fit(df)
    ev,_ = fa.get_eigenvalues()
    plt.scatter(range(1,df.shape[1]+1),ev)
    plt.plot(range(1,df.shape[1]+1),ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigen Value')
    plt.grid()
    
    fa = FactorAnalyzer(n_factors=num,rotation='varimax')
    factor_df=pd.DataFrame(fa.fit(np.nan_to_num(df)).transform(np.nan_to_num(df)),columns=['Component-'+str(idx+1) for idx in range(num)])
    loadings=pd.DataFrame(fa.loadings_,index=df.columns)
    print(loadings)
    
    return loadings,factor_df
    
    
    

# Define function to compare variable distributions between train and test

def distComparison(train, test):
    a = len(train.columns)
    if a%2 != 0:
        a += 1
    
    n = np.floor(np.sqrt(a)).astype(np.int64)
    
    while a%n != 0:
        n -= 1
    
    m = (a/n).astype(np.int64)
    coords = list(itertools.product(list(range(m)), list(range(n))))
    
    numerics = train.select_dtypes(include=np.number).columns
    cats = train.select_dtypes(exclude=np.number).columns
    
    fig = plt.figure(figsize=(15, 15))
    axes = gs.GridSpec(m, n)
    axes.update(wspace=0.25, hspace=0.25)
    
    for i in range(len(numerics)):
        x, y = coords[i]
        ax = plt.subplot(axes[x, y])
        col = numerics[i]
        sns.kdeplot(train[col].dropna(), ax=ax, label='train').set(xlabel=col)
        sns.kdeplot(test[col].dropna(), ax=ax, label='test')
        
    for i in range(0, len(cats)):
        x, y = coords[len(numerics)+i]
        ax = plt.subplot(axes[x, y])
        col = cats[i]

        train_temp = train[col].value_counts()
        test_temp = test[col].value_counts()
        train_temp = pd.DataFrame({col: train_temp.index, 'value': train_temp/len(train), 'Set': np.repeat('train', len(train_temp))})
        test_temp = pd.DataFrame({col: test_temp.index, 'value': test_temp/len(test), 'Set': np.repeat('test', len(test_temp))})

        sns.barplot(x=col, y='value', hue='Set', data=pd.concat([train_temp, test_temp]), ax=ax).set(ylabel='Percentage')


   
   

def weighted_hamming(data):
    """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
        the values between point A and point B are different, else it is equal the relative frequency of the
        distribution of the value across the variable. For multiple variables, the harmonic mean is computed
        up to a constant factor.

        @params:
            - data = a pandas data frame of categorical variables

        @returns:
            - distance_matrix = a distance matrix with pairwise distance for all attributes
    """
    categories_dist = []
    
    for category in data:
        X = pd.get_dummies(data[category])
        X_mean = X * X.mean()
        X_dot = X_mean.dot(X.transpose())
        X_np = np.asarray(X_dot.replace(0,1,inplace=False))
        categories_dist.append(X_np)
    categories_dist = np.array(categories_dist)
    distances = hmean(categories_dist, axis=0)
    return distances


def distance_matrix(data, numeric_distance = "euclidean", categorical_distance = "jaccard"):
    """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
        - Continuous
        - Categorical
        For ordinal values, provide a numerical representation taking the order into account.
        Categorical variables are transformed into a set of binary ones.
        If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
        variables are all normalized in the process.
        If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.
        
        Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C 
        like other distance metrics provided by scipy.

        @params:
            - data                  = pandas dataframe to compute distances on.
            - numeric_distances     = the metric to apply to continuous attributes.
                                      "euclidean" and "cityblock" available.
                                      Default = "euclidean"
            - categorical_distances = the metric to apply to binary attributes.
                                      "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                      available. Default = "jaccard"

        @returns:
            - the distance matrix
    """
    possible_continuous_distances = ["euclidean", "cityblock"]
    possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
    number_of_variables = data.shape[1]
    number_of_observations = data.shape[0]

    # Get the type of each attribute (Numeric or categorical)
    is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
    is_all_numeric = sum(is_numeric) == len(is_numeric)
    is_all_categorical = sum(is_numeric) == 0
    is_mixed_type = not is_all_categorical and not is_all_numeric

    # Check the content of the distances parameter
    if numeric_distance not in possible_continuous_distances:
        print("The continuous distance " + numeric_distance + " is not supported.")
        return None
    elif categorical_distance not in possible_binary_distances:
        print("The binary distance " + categorical_distance + " is not supported.")
        return None

    # Separate the data frame into categorical and numeric attributes and normalize numeric data
    if is_mixed_type:
        number_of_numeric_var = sum(is_numeric)
        number_of_categorical_var = number_of_variables - number_of_numeric_var
        data_numeric = data.iloc[:, is_numeric]
        data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
        data_categorical = data.iloc[:, [not x for x in is_numeric]]

    # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
    # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
    # but the value are properly replaced
    if is_mixed_type:
        data_numeric.fillna(data_numeric.mean(), inplace=True)
        for x in data_categorical:
            data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
    elif is_all_numeric:
        data.fillna(data.mean(), inplace=True)
    else:
        for x in data:
            data[x].fillna(data[x].mode()[0], inplace=True)

    # "Dummifies" categorical variables in place
    if not is_all_numeric and not (categorical_distance == 'hamming' or categorical_distance == 'weighted-hamming'):
        if is_mixed_type:
            data_categorical = pd.get_dummies(data_categorical)
        else:
            data = pd.get_dummies(data)
    elif not is_all_numeric and categorical_distance == 'hamming':
        if is_mixed_type:
            data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
        else:
            data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()

    if is_all_numeric:
        result_matrix = cdist(data, data, metric=numeric_distance)
    elif is_all_categorical:
        if categorical_distance == "weighted-hamming":
            result_matrix = weighted_hamming(data)
        else:
            result_matrix = cdist(data, data, metric=categorical_distance)
    else:
        result_numeric = cdist(data_numeric, data_numeric, metric=numeric_distance)
        if categorical_distance == "weighted-hamming":
            result_categorical = weighted_hamming(data_categorical)
        else:
            result_categorical = cdist(data_categorical, data_categorical, metric=categorical_distance)
        result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
                               number_of_categorical_var) / number_of_variables for j in range(number_of_observations)] for i in range(number_of_observations)])

    # Fill the diagonal with NaN values
    np.fill_diagonal(result_matrix, np.nan)

    return pd.DataFrame(result_matrix)


def knn_impute(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
               categorical_distance="jaccard", missing_neighbors_threshold = 0.5):
    """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
        attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
        remains missing. If there is a problem in the parameters provided, returns None.
        If to many neighbors also have missing values, leave the missing value of interest unchanged.

        @params:
            - target                        = a vector of n values with missing values that you want to impute. The length has
                                              to be at least n = 3.
            - attributes                    = a data frame of attributes with n rows to match the target variable
            - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                              value between 1 and n.
            - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                              Default = "mean"
            - numeric_distances             = the metric to apply to continuous attributes.
                                              "euclidean" and "cityblock" available.
                                              Default = "euclidean"
            - categorical_distances         = the metric to apply to binary attributes.
                                              "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                              available. Default = "jaccard"
            - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                              the correct value. Default = 0.5

        @returns:
            target_completed        = the vector of target values with missing value replaced. If there is a problem
                                      in the parameters, return None
    """

    # Get useful variables
    possible_aggregation_method = ["mean", "median", "mode"]
    number_observations = len(target)
    is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

    # Check for possible errors
    if number_observations < 3:
        print("Not enough observations.")
        return None
    if attributes.shape[0] != number_observations:
        print("The number of observations in the attributes variable is not matching the target variable length.")
        return None
    if k_neighbors > number_observations or k_neighbors < 1:
        print("The range of the number of neighbors is incorrect.")
        return None
    if aggregation_method not in possible_aggregation_method:
        print("The aggregation method is incorrect.")
        return None
    if not is_target_numeric and aggregation_method != "mode":
        print("The only method allowed for categorical target variable is the mode.")
        return None

    # Make sure the data are in the right format
    target = pd.DataFrame(target)
    attributes = pd.DataFrame(attributes)

    # Get the distance matrix and check whether no error was triggered when computing it
    distances = distance_matrix(attributes, numeric_distance, categorical_distance)
    if distances is None:
        return None

    # Get the closest points and compute the correct aggregation method
    for i, value in enumerate(target.iloc[:, 0]):
        if pd.isnull(value):
            order = distances.iloc[i,:].values.argsort()[:k_neighbors]
            closest_to_target = target.iloc[order, :]
            missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
            # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
            if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                continue
            elif aggregation_method == "mean":
                target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            elif aggregation_method == "median":
                target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            else:
                target.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

    return target
   
    
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score

def my_KMeans(data,n=5):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import MinMaxScaler
        df=data.select_dtypes(include=np.number)
        mean_values= np.round(df.mean(axis=0))
        sd_values=np.round(df.std(axis=0))
       
        flag1=1 if (mean_values.max()==0) & (sd_values.min()==1) else 0
        flag2=1 if (df.max(axis=0).max()==1) & (df.min(axis=0).min()==0) else 0
        if flag1 + flag2==0:
            
            
            scaler = MinMaxScaler()
            print('Data is not scaled. Applying MinMax Scaling.')
            df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
          
        Sum_of_squared_distances = []
        sil=[]
        K = range(2,n+1)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df)
            labels = km.labels_
            Sum_of_squared_distances.append(km.inertia_)
            sil.append(silhouette_score(df, labels, metric = 'euclidean'))
            
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        plt.plot(K, sil, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhoutte Score')
        plt.title('Silhoutte Method For Optimal k')
        plt.show()

        y_predict= KMeans(n_clusters=n).fit_predict(df)
        return y_predict
   

        
# from KMedoids import *
# import matplotlib.pyplot as plt

# def myKMedoids(data_,n=5):

 
#     n_clusters = range(2,n+1)
#     print('\n The Partitioning based clustering will start.Make sure the categorical variables have been label encoded.')
#     k_medoids = [KMedoids(n_cluster=i) for i in n_clusters]
#     k_medoids = [k_medoid.fit(data_) for k_medoid in k_medoids]
#     loss = [k_medoid.calculate_distance_of_clusters() for k_medoid in k_medoids]

#     # Plot elbow curve (to know best cluster count)
#     plt.figure(figsize=(13,8))
#     plt.plot(n_clusters,loss)
#     plt.xticks(n_clusters)
#     plt.xlabel('Number of Clusters')
#     plt.ylabel('Loss')
#     plt.title('Loss Vs No. Of clusters')
#     plt.show()
#     return k_medoids

