from sklearn import svm,datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data[:, :2]
y = (iris.target != 0)*1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

clf = svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')

#clf.fit(x_train,y_train)
clf.fit(x_train,y_train.ravel())

result = clf.predict(x_test)
print('Accuracy:',clf.score(x_test,y_test))