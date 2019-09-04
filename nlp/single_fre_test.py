"""
统计一段文字当中，每个字出现的频率，并按照排序输出。上限10个。
"""

import jieba

# 读取文件
read_path = '../dataset/dahuajiao.txt'
file = open(read_path, 'r')
str_input = file.read()
file.close()

# 文字处理，去除换行
str_input = str_input.replace('\n\n', '。')
print(str_input)

# 全模式
seg_list = jieba.lcut(str_input, cut_all=False)
print(seg_list)
# print("【全模式】：" + "/".join(seg_list))

# # 建立一个字典，没有出现的字就填进字典去
# chd = {}
# for ch in str_input:
#     if '\u4e00' <= ch <= '\u9fff':
#         if ch in chd:
#             chd[ch] += 1
#         else:
#             chd[ch] = 0
#
# list1 = sorted(chd.items(), key=lambda x:x[1], reverse=True)
# print(list1[:5])
