"""
文本处理
"""

# 读取文件
read_path = 'note/01.txt'
file = open(read_path, 'r')
file_read = file.read()
file.close()

# 去除空格和换行符
paper = file_read.replace('\n', '')
paper = paper.replace('　', '')

# 完成断句
punctuation = '！？，。“”（）：；、…'
paper_list = [paper]
for p in punctuation:
    new_list = []
    for str in paper_list:
        str_list = str.split(p)
        if '' in str_list:
            str_list.remove('')
        new_list += str_list
    paper_list = new_list

# 去除相同的句子
paper_list = list(set(paper_list))

# 只有一个字的句子就算了
for sen in paper_list:
    if len(sen) == 1:
        paper_list.remove(sen)

# l = len(paper_list)
# print(paper_list, l)

# 逐行写入文件
write_path = 'note/m1.txt'
file = open(write_path, 'w')
for line in paper_list:
    file.write(line)
    file.write('\n')
file.close()
