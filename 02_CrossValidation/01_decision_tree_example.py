import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/winequality-red.csv")
print(f'df head: {df.head()}')
print(f'df columns: {df.columns}')
print(f'df number of rows : {df.shape[0]}')


quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}

df['quality'] = df['quality'].map(quality_mapping)

## Use sample with frac=1 to SHUFFLE the ENTIRE dataset
df = df.sample(frac=1).reset_index(drop=True)
df_train = df.iloc[:1000, ]
df_test = df.iloc[1000:, ]

from sklearn import tree
from sklearn import metrics

cols  = list( set(df.columns) - set(['quality']) )
train_accuracy_list = []
test_accuracy_list = []
for max_depth in range(1, 26):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(df_train[cols], df_train['quality'])
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])
    train_accuracy = metrics.accuracy_score(df_train['quality'], train_predictions)
    test_accuracy = metrics.accuracy_score(df_test['quality'], test_predictions)
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

plt.plot(train_accuracy_list,label='train_accuracy')
plt.plot(test_accuracy_list, label='test_accuracy')
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()





