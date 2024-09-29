import pandas as pd
import numpy as np
# 读取 'data_mag.csv' 文件，没有标题行
testdata=pd.read_csv('data_mag.csv',header=None)
# 将 pandas DataFrame 转换为 numpy 数组
testdata=np.array(testdata)
testred=pd.read_csv('redshift.csv',header=None)
testred=np.array(testred)

#normalization
def scale_minmax(data):
    return (data-data.min())/(data.max()-data.min())

for i in range(0,15):
    testdata[:,i]=scale_minmax(testdata[:,i])

print(testdata[0])
print(testred.shape)
