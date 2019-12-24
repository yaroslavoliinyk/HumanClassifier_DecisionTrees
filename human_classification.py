import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

text_len = 70

def inner_print(text):
    remained_len = text_len - len(text)
    r_len1 = remained_len // 2 if remained_len % 2 == 0 else remained_len // 2 + 1
    r_len2 = remained_len // 2
    if (remained_len > 0):
        print('-' * r_len1, text, '-' * r_len2)
    else:
        print(text)

inner_print('Made by Yaroslav Oliinyk, 2019')
inner_print('Human classifier made with DECISION TREE ALGORITHM')
# Reading represented data
adult_data = pd.read_csv('adult.csv')
# We remove unnecessary columns and made X variables, and y - for prediction
X = adult_data.drop(['fnlwgt', 'education', 'capital.gain', 'capital.loss', 'income'], axis=1)
# We will predict our income
y = adult_data.income
# To use Decision tree algorithm, we should parse our data into so called 'dummy variables' 0 or 1
# e.g. We have column: Male or Female; get_dummies method will create 2 columns Male_col where 1
# will stand for male and 0 for Female and Female_col where 1s and 0s will be represented vice-versa
X = pd.get_dummies(X)
# fast splitting our set on test and train variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# to figure out the best tree depth, we will check depths from 1 to 15
max_depth_values = range(1, 15)
# DataFrame where we will store our results
scores_data = pd.DataFrame()
# checking what depth is the best
for max_depth in max_depth_values:
    # creating classifier
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    # write out test and train score
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    # MAKING CROSS-VALIDATION with 5 folds
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    # creating data score with results
    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score],
                                    'cross_val_score': [mean_cross_val_score]})
    # appending those results to the table
    scores_data = scores_data.append(temp_score_data)
inner_print('Results after teaching:')
print(scores_data)

max_depth = np.argmax(np.array(scores_data.cross_val_score))
print('Depth of the resulting tree will be:', max_depth)
print('Chosen best score after 5-cross validation in %:', scores_data.cross_val_score.max()*100, '%')
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
clf.fit(X_train, y_train)
print('Result after training',clf.score(X_test, y_test)*100, '5 of precision.')

# remaking data frame to make plot
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'], var_name='set_type', value_name='score')
# making plot
ax = sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
plt.show()
