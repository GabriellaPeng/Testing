import pandas as pd
import seaborn as sb

path = '/Users/gabriellapeng/PycharmProjects/pythonProject' + \
       "/data/titanic-training-data.csv"
data = pd.read_csv(path)
data.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

sum_nan = data.isna().sum()
# info_data = data.describe()

data.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
data.drop(data.index[0], inplace=True)

# sb.countplot(x='Survived', data=data)

# imputing missing data for age
# sb.boxplot(x='Parch', y='Age', data=data, palette='hls')
mean_group_data = data.groupby(data['Parch']).mean()


def impute_miss(cols):
    Age = cols[0]
    Parch = cols[1]
    if pd.isnull(Age):  # detect missing
        if Parch == 0:
            return mean_group_data['Age'][0]
        elif Parch == 1:
            return mean_group_data['Age'][1]
        elif Parch == 2:
            return mean_group_data['Age'][2]
        elif Parch == 3:
            return mean_group_data['Age'][3]
        elif Parch == 4:
            return mean_group_data['Age'][4]
        else:
            return mean_group_data['Age'].mean()

    else:
        return Age


data['Age'] = data[['Age', 'Parch']].apply(impute_miss, axis=1)

# encoder sex/embarked data; encoder cannot process nan data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sum_nan = data.isna().sum()
data.dropna(inplace=True)
data.reset_index(inplace=True)

label_encoder = LabelEncoder()
gender_dt = data['Sex']
gender_dt = label_encoder.fit_transform(gender_dt)
genger_df = pd.DataFrame(gender_dt, columns=['male_gender'])

emb_dt = data['Embarked']
emb_dt = label_encoder.fit_transform(emb_dt)

onehot_encoder = OneHotEncoder(categories='auto')
emb_dt = onehot_encoder.fit_transform(emb_dt.reshape(-1, 1))
emb_dt = emb_dt.toarray()
embarked_dF = pd.DataFrame(emb_dt, columns=['C', 'Q', 'S'])

data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
data_dmy = pd.concat([data, genger_df, embarked_dF], axis=1,
                     verify_integrity=True).astype(float)

# sb.heatmap(data_dmy.corr())
data_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)

# model
from sklearn.model_selection import train_test_split, cross_val_predict

x_train, x_test, y_train, y_test = train_test_split(data_dmy.drop(['Survived'], axis=1),
                                                    data_dmy['Survived'],test_size=0.2)

#depoly model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
y_pred = model.predict(x_test)

y_pred_cv = cross_val_predict(model, x_train, y_train, cv=5)

#evaluate model
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

classification_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)

confusion_matrix(y_pred_cv, y_pred)
precision_score(y_pred_cv, y_pred)

print()


