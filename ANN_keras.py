from keras.layers import *
from keras.models import Model

# 设置输入维度
input_dims = 15
inputs = Input(shape=(input_dims,))
#inputs=Flatten()(inputs)
# 构建模型结构
dense1 = Dense(32, activation='relu')(inputs)
dense2 = Dense(16, activation='relu')(dense1)
dense3 = Dense(1)(dense2)

# 定义模型
model = Model(inputs=inputs, outputs=dense3)

# 打印模型结构
print(model.summary())
