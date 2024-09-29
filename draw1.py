import pandas as pd
import numpy as np
import pylab as pl
from pythonProject2.dataClassify import pred, testY
'''将数组保存到文本文件中'''
np.savetxt('prediction-mag.csv', pred, delimiter=',', fmt='%s')#预测值
np.savetxt('redshift-mag.csv', testY, delimiter=',', fmt='%s')#真实值

res = pred - testY#误差
print('最小误差', min(res))
print('最大误差', max(res))

res = np.array(res)
print('误差组', res.shape)

'''分箱是一种将连续变量分为不同区间（或称为桶、箱）的方法，这在数据分析中很常见，特别是在准备数据进行可视化或建模时。
使用了 pandas 库中的 pd.cut 函数来将连续的数值型数组 res 分割成不同的区间（分箱），并计算每个区间的频数。'''
res = res.reshape(83186,)
bins = [-2.15,-2.05,-1.95,-1.85,-1.75,-1.65,-1.55,-1.45,-1.35,-1.25,-1.15,-1.05,-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45, 1.55,1.65,1.75,1.85,1.95,2.05,2.15]
cats = pd.cut(res, bins, right=True, duplicates='drop') #左开右闭

cats.value_counts()#计算每个箱的频数

#绘制直方图
def draw_hist(lenths):  # lenths 接受的其实是 sizeArry传来的数组 就是def get_data(lines) 返回的数据
    data = lenths

    # slice
    bins = [-2.15, -2.05, -1.95, -1.85, -1.75, -1.65, -1.55, -1.45, -1.35, -1.25, -1.15, -1.05, -0.95, -0.85, -0.75,
            -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
            1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15]

    pl.hist(data, bins)
    # x-axis
    pl.xlabel('Res')
    # y-axis
    pl.ylabel('Number')
    pl.title(' ')
    pl.show()

draw_hist(res)