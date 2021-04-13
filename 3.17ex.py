import pymysql

#db = pymysql.connect(host='localhost',user='root',password='root',port=3306)
#cursor = db.cursor()
#cursor.execute('SELECT VERSION()')
#data = cursor.fetchone()
#print('Database version:',data)
#cursor.execute("CREATE DATABASE spiders DEFAULT CHARACTER SET utf8")
#db.close()

def select():
    """查询"""
    try:
        #创建连接对象
        conn=pymysql.Connect(host='localhost',port=3306,db='python',user='root',passwd='root',charset='utf8')
        #创建可执行对象，可执行SQL语句
        cur=conn.cursor()
        #执行SQL语句,并传递参数
        cur.execute('select * from student where id=%s',[2])
        #查询一个结果
        result = cur.fetchone()
        print(result)
        #关闭
        conn.close()
    except  Exception as ex:
        print(ex)

def update():
    """修改"""
    try:
        #获取连接对象
        conn=pymysql.Connect(host='localhost',port=3306,db='python',user='root',passwd='root',charset='utf8')
        #创建可执行对象，可执行SQL语句
        cur=conn.cursor()
        #执行SQL语句，并传递参数
        count=cur.execute('update student set name=%s where id=%s',['张三',2])
        #判断结果
        if count>0:
            print('成功')
        else:
            print('失败')
        #提交
        conn.comit()
        #关闭
        conn.close()
    except Exception as ex:
        print(ex)

if __name__=='__main__':
    select()
    update()