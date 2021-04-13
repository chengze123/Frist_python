"""通过oop处理学生信息"""
# class Student(object):
# def __init__(self,name,age):
# self.name=name
# self.age=age

# def show(self):
# print('%s:%s'%(self.name,self.age))


# stu1=Student('Morty',22)
# stu2=Student('daisy',22)
# stu2.show()
# stu1.show()

"""继承"""
# 父类
# class Animal(object):
# def __init__(self,name):
# self.name=name

# def run(self):
# print('{} run...'.format(self.name))

# 子类
# class Cat(Animal):
# 重写父类方法
# def __init__(self,name,color):
# 调用父类的方法
# super().__init__(name)
# self.color=color
# def show(self):
# print('name={},color={}'.format(self.name,self.color))

# 子类
# class Dog(Animal):
# def run(self):
# print('{}run fast11...'.format(self.name))

# 创建对象
# cat=Cat('泡芙','白')
# 调用方法
# cat.run()
# cat.show()

# dog=Dog('汪汪')
# dog.run()

"""实现文件复制功能"""
# with open('D:\代码学习“十四五”，有“数”.txt','r',encoding='utf-8') as file:
# print(file.read())
# new_file=open('D:\练习.txt','w',encoding='utf-8')
# for i in file.readline():
# new_file.write(file.read())
# print('结束')

# file.close()
# new_file.close()

# 爬取猫眼电影前100
import requests
import re
import json
import time
import random
from requests.exceptions import  RequestException


#获取单一页面
def get_one_page(url):
   try:
       headers={
           'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
       }
       response=requests.get(url=url,headers=headers)
       if response.status_code==200:
           return  response.text
       return None
   except  RequestException:
       return None

#解析页面
def parse_one_page(html):
    pattern=re.compile(
        #获取排名
        r'<dd>.*?board-index.*?>(\d+?)</i>'
        #获取电影图片
        r'.*?data-src="(.*?)".*?</a>'
        #获取电影名称
        r'.*?class="name".*?title="(.*?)".*?</a>'
        #电影主演
        r'.*?class="star".+?(.*?)</p>'
        #发布时间
        r'.*?class="releasetime".+?(.*?)</p>'
        #评分
        r'.*?class="integer".+?(.*?)</i>'
        #评分
        r'.*?"fraction".+?(.*?)</i>.*?</dd>',re.S
    )
    re_lists=re.findall(pattern,html)
    for re_list in re_lists:
        yield{
            'index':re_list[0],
            'image':re_list[1],
            'title':re_list[2],
            'actor':re_list[3].strip()[3:],
            'time':re_list[4].strip()[5:],
            'score':re_list[5]+re_list[6]
        }

#获取所有的url
def url_list(offset):
    if offset==0:
        page_url='http://maoyan.com/board/4'
        return page_url
    else:
        page_url='http://maoyan.com/board/4'+'?offset='+str(offset)
        return page_url

#保存数据
def write(final_result):
    with open('爬取猫眼电影.txt','a',encoding='utf-8') as file:
        file.write(json.dumps(final_result,ensure_ascii=False)+'\n')

#主函数
def main():
        offset_list=[0,10,20,30,40,50,60,70,80,90]
        for offset in offset_list:
            url=url_list(offset)
            html=get_one_page(url)
            result=parse_one_page(html)
            for i in range(15):
                final_result=next(result)
                print(final_result)
                write(final_result)

if __name__=='__main__':
    main()
    time.sleep(1)