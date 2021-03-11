
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
def plot_matirix(data):
    scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='kde')
    plt.show()


def pre_process(df):
    # Data transformation
    # Convert categorical values to numeric using label encoder
    from sklearn import preprocessing
    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)

    # Encoding the categorical variable
    fit = df.select_dtypes(include=['object']).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))
    #Convert the categorical columns based on encoding
    for i in list(d.keys()):
        df[i] = d[i].transform(df[i].fillna('NA'))
    return(df)

# Define function for outlier\boxplot

def outlier_flag(data):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=numerics)
    print('Original data:'+str(data.shape))
    z = np.abs(stats.zscore(data))
    data=data[(z < 3).all(axis=1)]
    print('\nRemoval of outliers'+str(data.shape))
    return(data)

def box_plot(data):
    print(data.describe())
    sns.boxplot(data, orient="h",linewidth=2.5, palette="Set2")
    plt.show()


# Define function for bar plot by category

def hist_by_category(data,column_header):
	
	data.groupby(column_header).hist(alpha=0.4)


#Define function to impute missing data

def impute_values(df):
    #fancy impute removes column names.
    train_cols = df.columns
    # Use MICE to fill in each row's missing features
    df = pd.DataFrame(MICE(verbose=False).complete(df))
    df.columns = train_cols
    return(df)

def impute_all(data):
    categorical_columns = []
    numeric_columns = []
    for c in data.columns:
        if data[c].map(type).eq(str).any(): #check if there are any strings in column
            categorical_columns.append(c)
        else:
            numeric_columns.append(c)

    #create two DataFrames, one for each data type
    data_numeric = data[numeric_columns]
    data_categorical = pd.DataFrame(data[categorical_columns],columns=categorical_columns)
    print(numeric_columns)
    print(categorical_columns)


    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(missing_values=np.nan,max_iter=10, verbose=0)
    #only apply imputer to numeric columns
    data_numeric=pd.DataFrame(imp.fit_transform(data_numeric),columns=data_numeric.columns)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data_categorical=pd.DataFrame(imp.fit_transform(data_categorical),columns=categorical_columns,index=data_categorical.index)
    #join the two masked dataframes back together
    data_joined = pd.concat([data_numeric, data_categorical], axis = 1)
    return(data_joined)

def outlier_detection(df):
    #specify  column names to be modelled
       
    to_model_columns=df.select_dtypes([np.number]).columns
    from sklearn.ensemble import IsolationForest
    
    clf=IsolationForest(n_estimators=100,
                        max_samples='auto',
                         behaviour="new",
                        contamination=float(.10),
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
    print("\n Percentage of data points flagged as outliers ",str(pcnt)," %")
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
    return(df[df.anomaly==1][df.columns.difference(['anomaly'])])

def my_featureTransformToNormalDist(column,name,lamb=None):
    print('\nInitiating box cox transformation')
    list_lambda=list()
    
    original=col
    transformed_col, best_lambda = boxcox(df[col],lamb)
    sns.kdeplot(pd.DataFrame([original,transformed_col],columns=['Original','Transformed']),x="Box Cox transformation of "+str(name))
    print("\n Check for normality from the plot and store lambda value to apply to test data later")
    return transformed_col, best_lambda



def missing_values_table(df):
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

def under_sample(df,target):
    
    tl = TomekLinks(return_indices=True, ratio='majority')
    X_tl, y_tl, id_tl = tl.fit_sample(df[df.columns.difference([target])],df[target])

    print('Removed indexes:', id_tl)
    df1 = pd.DataFrame(X_tl)
    df1[target]=y_tl
    df1.head(10)
    return df1

def over_sample(df,target):
    smt = SMOTETomek(ratio='auto')
    columns=[col for col in df.columns if col not in target]
    
    X_smt, y_smt = smt.fit_sample(df[columns],df[target])
    X_smt=pd.DataFrame(X_smt, columns=columns)
    X_smt[target]=y_smt
    

    return(X_smt)

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
    
    numerics = train.select_dtypes(include=[np.number]).columns
    cats = train.select_dtypes(include=['category']).columns
    
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
   
    
    

