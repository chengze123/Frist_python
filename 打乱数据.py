import os
import random
#输出
out = open("train.txt",'w',encoding='utf-8')
lines=[]
#输入文件
with open("train1.txt", 'r',encoding='utf-8') as file:
	for line in file:
		lines.append(line)
random.shuffle(lines)
for line in lines:
	out.write(line)

file.close()
out.close()