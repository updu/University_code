import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import LinearSVC

# get the dataset
faces = fetch_olivetti_faces()
X = (faces.data*255).astype(int)
y = faces.target

# create a test set and a training set
idx = np.arange(len(X))
np.random.shuffle(idx)
train = idx[:2*len(X)/3]
test = idx[2*len(X)/3:]

# create the models
lda = LatentDirichletAllocation(n_topics=10)
svm = LinearSVC(C=10)

# evaluate everything
lda.fit(X[train])
T = lda.transform(X)
print(svm.fit(T[train], y[train]).score(T[test], y[test]))