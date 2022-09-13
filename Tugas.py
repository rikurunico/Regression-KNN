from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score


# Go Out = 0
# Stay Home = 1

X_train = np.array([
    [26, 8],
    [26, 11],
    [29, 8],
    [34, 10],
    [36, 11],
    [33, 8],
    [30, 9],
    [27, 10],
    [27, 11],
    [26, 11],
    [31, 13],
    [34, 8]
])

y_train = ['Go Out', 'Go Out', 'Go Out', 'Stay at Home', 'Stay at Home',
           'Go Out', 'Go Out', 'Go Out', 'Go Out', 'Go Out', 'Stay at Home', 'Go Out']

X_test1 = np.array([[30, 10]])
X_test2 = np.array([28, 13])
X_test3 = np.array([29, 9])
X_test4 = np.array([31, 12])
X_test5 = np.array([27, 8])

x_test_2 = np.array([
    [30, 10],
    [28, 13],
    [29, 9],
    [31, 12],
    [27, 8]
])

y_test = ['Stay at Home', 'Go Out', 'Go Out', 'Stay at Home', 'Go Out']

dis = np.sqrt(np.sum((X_train - X_test1)**2, axis=1))
print("\nTesting Datatest 1\n", dis)

dis2 = np.sqrt(np.sum((X_train - X_test2)**2, axis=1))
print("\nTesting Datatest 2\n", dis2)

dis3 = np.sqrt(np.sum((X_train - X_test3)**2, axis=1))
print("\nTesting Datatest 3\n", dis3)

dis4 = np.sqrt(np.sum((X_train - X_test4)**2, axis=1))
print("\nTesting Datatest 4\n", dis4)

dis5 = np.sqrt(np.sum((X_train - X_test5)**2, axis=1))
print("\nTesting Datatest 5\n", dis5)

nn_sort = np.argsort(dis)[:3]
print("\nSorting Datatest 1\n", nn_sort)

nn_sort2 = np.argsort(dis2)[:3]
print("\nSorting Datatest 2\n", nn_sort2)

nn_sort3 = np.argsort(dis3)[:3]
print("\nSorting Datatest 3\n", nn_sort3)

nn_sort4 = np.argsort(dis4)[:3]
print("\nSorting Datatest 4\n", nn_sort4)

nn_sort5 = np.argsort(dis5)[:3]
print("\nSorting Datatest 5\n", nn_sort5)

nn_goStay = np.array(y_train)[nn_sort][:3]
print("\nGo Out or Stay at Home Datatest 1\n", nn_goStay)

nn_goStay2 = np.array(y_train)[nn_sort2][:3]
print("\nGo Out or Stay at Home Datatest 2\n", nn_goStay2)

nn_goStay3 = np.array(y_train)[nn_sort3][:3]
print("\nGo Out or Stay at Home Datatest 3\n", nn_goStay3)

nn_goStay4 = np.array(y_train)[nn_sort4][:3]
print("\nGo Out or Stay at Home Datatest 4\n", nn_goStay4)

nn_goStay5 = np.array(y_train)[nn_sort5][:3]
print("\nGo Out or Stay at Home Datatest 5\n", nn_goStay5)

b = Counter(np.take(y_train, dis.argsort()[:3]))
print("\nMost Common In Datatest 1\n", b.most_common(1)[0][0])

b2 = Counter(np.take(y_train, dis2.argsort()[:3]))
print("\nMost Common In Datatest 2\n", b2.most_common(1)[0][0])

b3 = Counter(np.take(y_train, dis3.argsort()[:3]))
print("\nMost Common In Datatest 3\n", b3.most_common(1)[0][0])

b4 = Counter(np.take(y_train, dis4.argsort()[:3]))
print("\nMost Common In Datatest 4\n", b4.most_common(1)[0][0])

b5 = Counter(np.take(y_train, dis5.argsort()[:3]))
print("\nMost Common In Datatest 5\n", b5.most_common(1)[0][0])

lb = LabelBinarizer()
y_trainBin = lb.fit_transform(y_train)
print("\nLabel Binarizer Training\n", y_trainBin.T[0])

y_testbin = lb.fit_transform(y_test)
print("\nLabel Binarizer Testing\n", y_testbin.T[0])

k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_trainBin.reshape(-1))
pred_bin = clf.predict(x_test_2)
pred_label = lb.inverse_transform(pred_bin)
print("\nPrediction Datatest\n", pred_label)

print("\nAccuracy Score\n", accuracy_score(y_test, pred_label))
print("\nRecall Score\n", recall_score(y_testbin, pred_bin))
print("\nPrecision Score\n", precision_score(y_testbin, pred_bin))
