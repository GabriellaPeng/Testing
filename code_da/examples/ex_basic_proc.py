import pandas as pd

titanic = pd.read_csv('../../../data/titanic.csv',
                      usecols=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
titanic.head()

# Drop all categorical features
cat_feat = ['PassengerId', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
titanic.drop(cat_feat, axis=1, inplace=True)
titanic.head()


# Explore Continuous Featuresc
# Look at the general distribution of these features
titanic.describe()

# Look at the correlation matrix
titanic.corr()

# Look at fare by different passenger class levels
titanic.groupby('Pclass')['Fare'].describe()