"""
将训练过的特征值，存入数据库中
"""

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_mnist.settings') 

import django
django.setup()

import json
import datetime
import numpy as np
from orm.models import DataSet, Character
from orm.train_character import get_character


def get_dataset():
    """返回数据集实体类，如没有就创建"""

    qs = DataSet.objects.filter(name='mnist')
    if len(qs):
        return qs[0]
    else:
        dataset = DataSet()
        dataset.name = 'mnist'
        dataset.training_sum = 60000
        dataset.recognize_sum = 10000
        dataset.save()
        return DataSet.objects.get(name='mnist')
        

def main():
    # 定义要训练的标签
    for label in range(10):
        # 先检查一下这个标签是否训练过了
        qs = Character.objects.filter(label=str(label), dataset__name='mnist')
        if len(qs):
            continue
        
        # 定义开始时间
        begin_time = datetime.datetime.now()
        
        # 训练特征值
        w, b = get_character(label)

        # 定义结束时间
        end_time = datetime.datetime.now()

        # 将numpy数组转换成json
        w_l = w.tolist()
        w_s = json.dumps(w_l)

        # 存入models
        character = Character()
        character.dataset = get_dataset()
        character.label = label
        character.w = w_s
        character.b = b
        character.begin_time = begin_time
        character.end_time = end_time
        character.save()

if __name__ == '__main__':
    main()