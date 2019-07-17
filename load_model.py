"""
调用识别函数，输出识别率
"""

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_mnist.settings') 

import django
django.setup()

import json
import numpy as np
from orm.models import DataSet, Character
from orm.recognize import recognize_single, recognize_rank


def single():
    """单边识别实验"""

    # 对要识别的标签进行循环
    for label in range(10):
        # 得到特征值
        character = Character.objects.get(label=str(label), dataset__name='mnist')
        w_s = character.w
        w_l = json.loads(w_s)
        w = np.array(w_l)
        b_s = character.b
        b = float(b_s)

        # 调用识别函数，开始识别
        rate_c, rate_w = recognize_single(label, w, b)

        print('label = {}, rate_c = {}, rate_w = {}'.format(label, rate_c, rate_w))


def rank():
    """rank识别实验"""

    chatacter_dict = {}
    qs = Character.objects.filter(dataset__name='mnist')
    for character in qs:
        label = int(character.label)
        w_s = character.w
        w_l = json.loads(w_s)
        w = np.array(w_l)
        b_s = character.b
        b = float(b_s)
        chatacter_dict[label] = [w, b]

    correct_recognize_rate = recognize_rank(chatacter_dict)
    print('correct_recognize_rate =', correct_recognize_rate)

if __name__ == '__main__':
    rank()
