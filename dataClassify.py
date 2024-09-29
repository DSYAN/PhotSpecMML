"""使用 Keras 构建和训练神经网络模型的示例，用于回归任务。代码中包含了数据预处理、模型构建、编译和训练的步骤"""
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
from ANN_keras import model
from preprocess_data import testdata, testred

# 数据集切分
(trainX, testX, trainY, testY) = train_test_split(testdata,testred, test_size=0.2, random_state=42)

# 转换标签为one-hot encoding格式
#lb = LabelBinarizer()
#trainY = lb.fit_transform(trainY)
#testY = lb.transform(testY)

callbacks = [EarlyStopping(patience=3), CSVLogger('model/mag.csv'),
             ModelCheckpoint('model/tesstmag.keras', save_best_only=True,save_weights_only = False)]

model.compile(loss='mean_absolute_error',optimizer=Adam(1e-3, amsgrad=True), metrics=['mse'])

H = model.fit(trainX,trainY, epochs=100,validation_data = (testX,testY),
                      batch_size = 256, callbacks=callbacks)

print('testY:', testY)

'''调用模型对测试特征进行预测，结果存储在pred中'''
pred = model.predict(testX)
print('pred:', pred)