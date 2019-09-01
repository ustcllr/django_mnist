"""
统计一段文字当中，每个字出现的频率，并按照排序输出。上限10个。
"""

str1 = '太阳出来我爬山坡，爬到了山顶我想唱歌，歌声飘给我妹妹听啊，听到我歌声她笑呵呵。'

ch = []
for zi in str1:
    if '\u4e00' <= zi <= '\u9fff':
        ch.append(zi)

# 建立一个字典，没有出现的字就填进字典去
chd = {}
for zi in ch:
    if zi in chd:
        chd[zi] += 1
    else:
        chd[zi] = 0

list1 = sorted(chd.items(), key=lambda x:x[1], reverse=True)
print(list1[:5])
