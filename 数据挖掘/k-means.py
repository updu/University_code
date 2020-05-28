import numpy as np
import matplotlib.pyplot as plt

data1=np.random.normal(10.0,10.0,[100,2])
data2=np.random.normal(40.0,10.0,[100,2])
data3=np.random.normal(80.0,10.0,[100,2])
data4=np.random.normal(130.0,10.0,[100,2])
data=np.vstack((data1 ,data2,data3,data4))
print("data:",data)

#计算数据到每个centers的距离，然后存在dist中
def distance(data,centers):
    dist = np.zeros((data.shape[0],centers.shape[0]))
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i,j] = np.sqrt(np.sum((data[i]-centers[j])**2))
    return dist

#输出每个点属于的类
def near_center(data,centers):
    dist = distance(data,centers)
    near_cen = np.argmin(dist,1)
    print("near_cen:",near_cen)
    return near_cen

def kmeans(data,k):
    centers = np.random.choice(np.arange(-5,10,0.1),(k,2))  #随机选取k行2列的矩阵
    print(centers)
    for _ in range(10): #循环十次
        near_cen = near_center(data,centers)    #near_cen为每个点属于的类别（一行四百列）
        for ci in range(k):
            centers[ci] = data[near_cen == ci].mean()
    return  centers,near_cen

centers,near_cen = kmeans(data,4)
print(data)
plt.scatter(data[:,0],data[:,1],c=near_cen)
plt.show()