import numpy as n#从文本文件中读取数据
with open(r'data.txt','r',encoding='utf-8') as f:
    contents=f.read()
    a = [i for i in contents.split(',')]
    data = list(map(int, a))p


# print(len(data))

#从键盘输入箱的深度
print("请输入箱的深度：")
depth = input()
# print(depth)

Threshold = 10 #阈值
data_new = []

#对数据预处理：若分箱时末箱数据不足，则用最后之值复制代替。
def data_process(data,depth):

    data_num = len(data)
    val = data_num % int(depth)

    #假如分箱时末箱数据不足时执行条件语句
    if(val != 0):
        data_num_new = (int(data_num/int(depth))+1)*int(depth)
        data1 = data

        for x in range(data_num,data_num_new):
            data1.append(data[len(data)-1])
    else:
    #分箱时末箱数据充足
        data1 = data

    return data1

#对数据进行分箱
def split_data(data,depth):

    #获得预处理之后的数组
    data_new = data_process(data,depth)
    data_val = []
    for i in range(0,len(data_new),int(depth)):        #分割
        data_val.append(data_new[i:i + int(depth)])

    return data_val

#获得分箱后的数据
val = split_data(data,depth)
# print(val)

#离群点
for i in range(len(val)):
    ave = np.array([np.mean(val[i])]*int(depth))
    x = val[i] - ave
    for n in range(int(depth)):
        if(abs(x[n])>Threshold):     #阈值设置为10
            print("离群点：",val[i][n])
#按箱平均值平滑法
print("按箱平均值平滑法")
for i in range(len(val)):
    ave = np.array([np.mean(val[i])]*int(depth))
    print(ave)

#按箱中值平滑法
print("按箱中值平滑法")
for i in range(len(val)):
    med= np.array([np.median(val[i])]*int(depth))
    print(med)


#按箱边界值平滑法
d=[]
print("按箱边界值平滑法")
for i in range(len(val)):
    tmp=val[i].copy()
    #print(tmp)
    for j in range(int(depth)):
        if abs(tmp[j]-tmp[0])>abs(tmp[j]-tmp[-1]):       #判断与边界的距离
                tmp[j]=tmp[-1]
        else:
                tmp[j]=tmp[0]
    d.append(tmp)
print(d)


#按箱中列数平滑法
print("按箱中列数平滑法")
for i in range(len(val)):
    g = (val[i][0]+val[i][-1])/2
    lie= np.array([g]*int(depth))
    print(lie)