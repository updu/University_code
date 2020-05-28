import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Load data
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

_, h, w = lfw_dataset.images.shape

X = lfw_dataset.data
y = lfw_dataset.target  #五类：0，1，2，3，4，5
target_names = lfw_dataset.target_names

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compute a PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
#fit(X)，表示用数据X来训练PCA模型
#   函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。

# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#用X来训练PCA模型，同时返回降维后的数据。
#列如newX=pca.fit_transform(X)，newX就是降维后的数据。

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
# print(y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))
# print("y_test",y_test)

# precision = 0
# for i in range(0, len(y_test)):
#     if (y_test[i] == y_pred[i]):
#         precision = precision + 1
#
# precision_avg = precision / len(y_test)
# print(precision_avg)






























# precision = 0
# precision_average = 0.0
# kf = KFold(n_splits=10, shuffle=True)
# for train, test in kf.split(X_train_pca):
#     clf = clf.fit(X_train_pca, y_train)
#     # print(clf.best_estimator_)
#     test_pred = clf.predict(X_test_pca)
#     for i in range(0, len(X_test_pca)):
#         if (y_test[i] == test_pred[i]):
#             precision = precision + 1
#     precision_average = precision_average + float(precision) / len(y_test)
# precision_average = precision_average / 10
# print(precision_average)
# print(classification_report(y_test, y_pred, target_names=target_names))