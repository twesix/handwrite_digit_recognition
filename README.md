# 手写数字识别demo

借助于keras训练了一个识别手写数字的模型，在canvas上写下数字，会把数字转换成符合模型输入的格式，然后由模型进行识别。

canvas的大小是280 x 280， 模型的输入是28 x 28，canvas中的图像经过压缩和处理之后由神经网络模型进行识别。

### 依赖

运行环境：python3

1. keras(依赖于tensorflow)
2. flask

### 文件说明

#### index.html

用来演示的html文件，内部含有处理canvas图像的过程

#### mnist.npz

用来训练模型的数据

#### model.py

加载训练好的模型

#### model_digit.h5

训练好的模型的权值

#### model_digit.json

训练好的模型的结构

#### server.py

开启一个简单的flask服务器

#### train_the_model.py

训练模型并保存

#### trained_model.h5

包含训练好的模型的结构和权值以及训练状态等信息