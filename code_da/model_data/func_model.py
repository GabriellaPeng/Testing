from collections import Counter

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import neighbors, datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, \
    recall_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale


# load
def load_csv_to_df(file=None, colnames=None):
    df = pd.read_csv(f'{file}', names=colnames)
    return df


# transform/extract
# connect sql.py

# process
def find_nans(df):
    '''df[colname].isna() | df[colname].notna()'''
    mask_nums = df.isna().any()  # 每个col的是否有nan
    sum_nan = df.isna().sum()  # count missing values by columns 每个col的nan的数量
    total_nans = df.isna().sum().sum()  # count totall missing values
    return {'nans by column': mask_nums, 'Sum nans by column': sum_nan,
            'total nums': total_nans}


# Process - Data label encoder; convert categorical variables to dummy indicators
def label_encoder(df, colname, plot=False):
    '''colname | ['name1', 'name2']'''
    Label_encoder = LabelEncoder()
    df = Label_encoder.fit_transform(df['col'])

    Binary_encoder = OneHotEncoder(categories='auto')
    BE = Binary_encoder.fit_transform(df['col'].reshape(-1, 1))
    df_BE = pd.DataFrame(BE, columns=['a', 'b', 'c'])  # 假设原始df里的'col'有三个label: a, b, c
    df = pd.concat([df, df_BE], axis=1, verify_integrity=True)

    if plot:
        sb.boxplot(x=colname, data=df)
        # sb.boxplot(x=col1, y=col2, data=df)
    return df


# Process - groupby
def groupdata(df, colnames):
    '''colname | ['name1', 'name2']'''
    groups = df.groupby(df[colnames])
    return groups.mean()


# Process - Check correlations
def check_correlations(df, drop=False):
    corr = df.corr()
    print(corr)
    sb.heatmap(corr)
    sb.pairplot(df)
    if drop:
        df.drop(['col1', 'col2'], axis=1, inplace=True)  # Drop correlation low的变量


# Model
# split
def split_data(df, data_for_split, target_data, test_size, standardize=False):
    X_train, X_test, y_train, y_test = train_test_split(
        data_for_split, target_data, test_size=test_size)

    if standardize:
        S_x_test, S_x_train = scale(X_test), scale(X_train)


# Deploy model
def deploy_supervised_model(model_type, X_train, X_test, y_train, cross_validate=False):
    '''linearReg | X, y =X_train, y_train '''
    if model_type == 'linear Regression':
        model = LinearRegression(normalize=True).fit(X_train, y_train)
        score, coef, intercept = model.score(X_train,
                                             y_train), model.coef_, model.intercept_
        print('regression score', score, 'coef', coef, 'intercep', intercept)

    elif model_type == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear').fit(X_train,
                                                           y_train)  # 'liblinear'仅限于单对单方案

    elif model_type == 'NN':
        '''scale X_test, X_train'''
        model = Perceptron(max_iter=50, eta0=0.15, tol=1e-3).fit(X_train,
                                                                 y_train.ravel())
    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train.ravel())

    elif model_type == 'KNN':
        '''scale X_test, X_train'''
        model = neighbors.kNeighborsClassifier().fit(X_train, y_train)
    elif model_type == 'Bayes':
        '''scale X_test, X_train'''
        model = BernoulliNB(binarize=0.1).fit(X_train, y_train)  # or binarize=True

        # model = MultinomialNB().fit(X_train, y_train)
        # model = GaussianNB().fit(X_train, y_train)
    if cross_validate:
        y_pred = cross_val_predit(model, x_train, y_train,
                                  cv=5)  # K-fold cross validation
    y_pred = model.predit(X_test)

    return y_pred, model


def deploy_unsupervised_DR_model(model_type, data, variable_names):
    '''data is 2 dims array | shape (150, 4) -4 columns with 150 rows
    comps:  ~ -1 or 1 = the factor has a strong influence on the variable.
			~ 0 = factor weakly influences the variable.
            > 1 = highly correlated factors.'''
    if model_type == 'FA':
        model = FactorAnalysis().fit(data)

    elif model_type == 'PCA':
        model = PCA().fit_transform(data)
        print('explained ratio',
              model.explained_variance_ratio_)  # 留取大头a + b + c+ d=1;
        # Explained variance ratio: how much information is compressed into the four components.
    comps = pd.DataFrame(model.components_, columns=variable_names)
    sb.heatmap(comps, annot=True)

    return model, comps


def deploy_Kmeans_cluster_model(n_clusters, data, plot=False):
    '''unsupervised clustering'''
    model = KMeans(n_clusters=n_clusters, random_state=5).fit(data)
    klabels = model.labels_  # used as color in plotting
    if plot:
        plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=klabels,
                    s=50)
        plt.title('K-Means Classification')


def deploy_Hclustering_model(n_clusters, data, plot=False):
    '''unsupervised clustering, X|(32,4); y|32'''
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                    linkage='ward').fit(data)
    # affinity=manhattan/cosine; linkage=average/complete

    if plot:
        Z = linkage(data, 'ward')  # ward is a linkage method
        dn = dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45.,
                        leaf_font_size=15, show_contracted=True)
        # The p parameter for truncate_mode. Truncation is used to condense the dendrogram
        plt.title('Truncated Hierarchial Clustering Diagram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        # plt.axhline(y=500)
        # plt.axhline(y=150)
        plt.show()


def deploy_DBSCan(data, plot=False):
    '''define outliers'''
    model = DBSCAN(eps=0.8, min_samples=19).fit(data)
    outliers_df = pd.DataFrame(data)
    outliers_df = outliers_df[model.labels_ == -1]  # return row index values for each of
    # those outlier records.
    count = Counter(model.labels_)
    if plot:
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, 1, 1])  # l,b,w,h

        ax.scatter(data[:, 2], data[:, 1], c=model.labels_, s=120)
        # ax.set_xlabel('x_label')
        # ax.set_ylabel('y_label')
        plt.title('DBSCAN for Outlier Detection')


# Model evaluation
def model_evaluation(y_test, y_pred, type):
    if type == 'classification_report':
        report = classification_report(y_test, y_pred)
    elif type == 'confusion_matrix':
        report = confusion_matrix(y_test, y_pred)
    elif type == 'precision_score':
        report = precision_score(y_test, y_pred)
    elif type == 'recall_score':
        report = recall_score(y_test, y_pred)
    return report


print()
