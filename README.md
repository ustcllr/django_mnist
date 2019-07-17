- 简介：本项目通过基于成本函数的梯度下降法，训练mnist的特征值w和b。训练方法是对每一个标签都进行监督学习，然后分别使用单边的方法和rank的方法，去识别测试集中的数字。训练次数为1000次，机器学习率为10。

- 项目分两步，第一步是计算模型，把特征值都存放在sqlite3中，然后可以跑识别函数，输出识别率。单边识别率很低，内容如下：

  > label = 1, rate_c = 0.97, rate_w = 0.99
  >
  > label = 2, rate_c = 0.11, rate_w = 1.0
  >
  > label = 3, rate_c = 0.93, rate_w = 0.94
  >
  > label = 4, rate_c = 0.93, rate_w = 0.98
  >
  > label = 5, rate_c = 0.93, rate_w = 0.92
  >
  > label = 6, rate_c = 0.9, rate_w = 0.99
  >
  > label = 7, rate_c = 0.88, rate_w = 0.99
  >
  > label = 8, rate_c = 0.22, rate_w = 1.0
  >
  > label = 9, rate_c = 0.97, rate_w = 0.8

- 这是一个Django项目。数据集需要存放在项目中，可使用自动下载程序。