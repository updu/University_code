#coding:utf8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
    img=[]
    for i in range(40):
        for j in range(10):
            path='orl_faces\\s'+str(i+1)+'\\'+str(j+1)+'.pgm'
            a=cv2.imread(path,0)
            a=a.flatten()/255.0
            img.append(a)
    return img

def dis(A,B,dis_type=0,s=None):
    if dis_type==1:  # 欧式距离
        return np.sum(np.square(A-B))
    elif dis_type==2:  # 马式距离
        f=np.sqrt(abs(np.dot(np.dot((A-B),s.I),(A-B).T)))  # h增大时右侧会出现负值,防止溢出可以s/np.linalg.norm(s)
        return f.tolist()[0][0]
    else:  # 曼哈顿距离
        return np.sum(abs(A-B))

def pca(data,h,dis_type=0):
    q,r=np.linalg.qr(data.T)
    u,s,v=np.linalg.svd(r.T)
    fi=np.dot(q,(v[:h]).T)
    y=np.dot(fi.T,data.T)
    ym=[np.mean(np.reshape(x,(40,10)),axis=1) for x in y]
    ym=np.reshape(ym,(h,40))
    c=[]
    if dis_type==2:# 计算马氏距离的额外处理"
        yr=[np.reshape(x,(40,10)) for x in y]
        yr=[[np.array(yr)[j][k] for j in  range(h)]for k in range(40)]
        for k in yr:
            k=np.reshape(k,(h,10))
            e=np.cov(k)
            c.append(e)
    return fi,ym,c

def validate(fi,ym,test,label,dis_type=0,c=None):
    ty=np.dot(fi.T,test.T)
    correctnum=0
    testnum=len(test)
    for i in range(testnum):
        if dis_type==2:
            n=len(ym.T)
            dd=[dis(ty.T[i],ym.T[n_],dis_type,np.mat(c[n_])) for n_ in range(n)]
        else:
            dd=[dis(ty.T[i],yy,dis_type) for yy in ym.T]
        if np.argsort(dd)[0]+1==label[i]:
            correctnum+=1
    rate = float(correctnum) / testnum
    print("Correctnum = %d, Sumnum = %d" % (correctnum, testnum), "Accuracy:%.2f" % (rate))
    return rate

if __name__ == '__main__':
    img=load_img()
    test=img
    label=[a+1 for a in range(40) for j in range(10)]
    index=list(range(400))
    np.random.shuffle(index)
    label_=[label[i] for i in index]
    test_=np.mat([test[i] for i in index])
    x_=[2**i for i in range(9)]
    d_=['Manhattan Distance','Euclidean Metric', 'Mahalanobis Distance']
    for j in range(3):
        y_=[]
        plt.figure()
        for i in range(9):
            fi,ym,c=pca(np.mat(img),h=x_[i],dis_type=j)
            y_.append(validate(fi,ym,test_,label_,dis_type=j,c=c))
        plt.ylim([0,1.0])
        plt.plot(x_,y_)
        plt.scatter(x_,y_)
        plt.xlabel('h')
        plt.ylabel('Accuracy')
        plt.title(d_[j])
    plt.show()
