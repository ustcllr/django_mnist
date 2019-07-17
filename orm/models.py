from django.db import models


class DataSet(models.Model):
    """数据集的信息"""

    name = models.CharField(max_length=50, verbose_name='名称')
    training_sum = models.IntegerField(verbose_name='训练集的数量')
    recognize_sum = models.IntegerField(verbose_name='识别集的数量')


class Character(models.Model):
    """根据训练集训练出来的特征值"""
    
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)
    label = models.CharField(max_length=50, verbose_name='目标标签')
    w = models.TextField(max_length=50000, verbose_name='向量特征值')
    b = models.CharField(max_length=50, verbose_name='常数特征值')
    begin_time = models.DateTimeField(verbose_name='训练开始时间')
    end_time = models.DateTimeField(verbose_name='训练结束时间')
    