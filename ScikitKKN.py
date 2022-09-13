from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])

y_train = ['male', 'male', 'male', 'male', 'female',
           'female', 'female', 'female', 'female']

X_test = np.array([[155, 70]])

lb = LabelBinarizer()
y_trainbin = lb.fit_transform(y_train)
print(y_trainbin)

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_trainbin.reshape(-1))
pred_bin = clf.predict(np.array([X_test]).reshape(1, -1))[0]
pred_label = lb.inverse_transform(pred_bin)
print(pred_label)

x_test = np.array([[168, 65],
                   [190, 96],
                   [160, 52],
                   [169, 67]])

y_test = ['male', 'male', 'female', 'female']

y_testbin = lb.transform(y_test)
print('label biner:%s ' % y_testbin.T[0])

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_trainbin.reshape(-1))
pred_bin = clf.predict(x_test)
pred_label = lb.inverse_transform(pred_bin)
print('pred label:%s ' % pred_label)

print('accuracy:%s ' % accuracy_score(y_testbin, pred_bin))
print('recall:%s ' % recall_score(y_testbin, pred_bin))
print('precision:%s ' % precision_score(y_testbin, pred_bin))
