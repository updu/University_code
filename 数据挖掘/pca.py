import numpy as np
from scipy import linalg

data=np.random.randint(5,size=(3,4))
print("data",data)

mean = sum(data)/3
data = data - mean
print("mean",mean)

cova = np.cov(data)
print("cova",cova)

eigvals,eigvectors = linalg.eig(cova)
print("eigvals",eigvals)
print("eigvectors",eigvectors)

print("w",eigvectors[:,0:2])



