from lxml import etree
import requests
import time
import random
import json
from queue import Queue

class DouBanSpider:
    """爬虫类"""
    def __init__(self):
        """构造方法"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
        }
        #baseUrl:基础URL
        self.baseUrl="https://movie.douban.com/top250"
        #dataQueue:队列存储数据
        self.dataQueue=Queue()
        #num数字编号
        self.num=1

    def loadPage(self,url):
        """向URL发送请求，获取响应内容"""
        #随机休眠0~1秒，避免爬虫过快导致爬虫被封
        time.sleep(random.random())
        return requests.get(url,headers=self.headers).content

    def parsePage(self,url):
        """根据起始URL提取所有的URL"""
        #获取URL对应的响应内容
        content=self.loadPage(url)
        #xpath处理得到对应的element对象
        html=etree.HTML(content)

        #所有的电影节点
        node_list=html.xpath("//div[@class='info']")
        for node in node_list:
            #每部电影的标题
            title=node.xpath(".//span[@class='title']/text()")[0]
            #每部电影的评分
            score=node.xpath(".//span[@class='rating_num']/text()")[0]
            #将数据存储到队列中
            self.dataQueue.put(score+"\t"+title)
        #只有在第一页时才获取所有的URL组成的列表，其他页就不再获取,paginator是页数div的class
        if url==self.baseUrl:
            return [self.baseUrl+link for link in html.xpath("//div[@class='paginator']/a/@href")]

    def startWork(self):
        """爬虫入口"""
        #第一个页面的请求，需要返回所有页面链接，并提取第一页的电影信息
        link_list=self.parsePage(self.baseUrl)
        #循环发送每个页面的请求，并获取所有电影信息
        for link in link_list:
            self.parsePage(link)
        #循环get队列的数据，直到队列为空再退出
        while not self.dataQueue.empty():
            print(self.num)
            print(self.dataQueue.get())
            self.num +=1
            self.save(self.dataQueue.get())

    def save(self,item):
        with open('.././douban.txt','a',encoding='utf-8') as file:
            print(11)
            file.write(item)
            file.write('\n')

if __name__=="__main__":
    #创建爬虫对象
    spider=DouBanSpider()
    #开始时间
    start=time.time()
    #开始爬虫
    spider.startWork()
    #结束时间
    stop=time.time()
    #打印结果
    print("\n[LOG]:%f seconds..." %(stop-start))
