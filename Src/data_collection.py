# 댓글을 달 빈 리스트를 생성합니다.
List = []
# 라이브러리를 로드합니다.
from bs4 import BeautifulSoup
import requests
import re
import sys
import pprint
import pandas as pd  
import sys, os
import os.path as osp
  
def to_csv_file(id_list,comment_list):   
        
    # dictionary of lists  
    dict = {'id':id_list,'text':comment_list}  
    
        
    df = pd.DataFrame(dict) 
    print(df)
        
    curdir = os.getcwd()
    path = osp.join(curdir,'..','data','comments_kor.csv')
    # saving the dataframe 
    df.to_csv(path, index=False)
 
 
# 여러 리스트들을 하나로 묶어 주는 함수입니다.
def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList

def collect_comments():
    result_id_list = [] 
    result_comment_list = [] 

    # 네이버 뉴스 url을 입력합니다.
    url2 = "https://news.naver.com/main/read.nhn?m_view=1&includeAllCount=true&mode=LSD&mid=shm&sid1=100&oid=422&aid=0000430957"
    url_list2 =  [
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001917670&rankingType=RANKING", 
    "https://news.naver.com/main/ranking/read.naver?m_view=1&includeAllCount=true&mode=LSD&mid=shm&sid1=001&oid=016&aid=0001918457&rankingType=RANKING", 
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001919505&rankingType=RANKING", 
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001917753&rankingType=RANKING", 
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001920612&rankingType=RANKING", 
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001917789&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001922773&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001920052&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001920757&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001923414&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001923417&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001922515&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=016&aid=0001924008&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?m_view=1&includeAllCount=true&mode=LSD&mid=shm&sid1=001&oid=016&aid=0001924180&rankingType=RANKING",
    "https://news.naver.com/main/read.naver?mode=LSD&mid=shm&sid1=105&oid=031&aid=0000642274",
    ]

    url_list = [
    "https://news.naver.com/main/read.naver?mode=LSD&mid=shm&sid1=100&oid=032&aid=0003117110",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=214&aid=0001166878&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=052&aid=0001678774&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=469&aid=0000647463&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=015&aid=0004642127&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=469&aid=0000647441&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=214&aid=0001166883&rankingType=RANKING",
    "https://news.naver.com/main/ranking/read.naver?mode=LSD&mid=shm&sid1=001&oid=448&aid=0000346757&rankingType=RANKING"
    ]
    
    for url in url_list:
        oid = url.split("oid=")[1].split("&")[0] #422
        aid = url.split("aid=")[1] #0000430957
        page = 1
        header = {
            "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
            "referer": url,
        }

        while True:
            c_url = "https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news" + oid + "%2C" + aid + "&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page=" + str(
                page) + "&refresh=false&sort=FAVORITE"
            # 파싱하는 단계입니다.
            r = requests.get(c_url, headers=header)
            cont = BeautifulSoup(r.content, "html.parser")

            total_comm = str(cont).split('comment":')[1].split(",")[0]
        
            comment_list = re.findall('"contents":([^\*]*),"userIdNo"', str(cont))
            #id_list = re.findall('"userName":([^\*]*),"userProfileImage"', str(cont))
            #id_list = re.findall('"userName":([^\*]*),"userProfileImage"', str(cont))
            id_list = re.findall(r'"userName":.+?,"userProfileImage"',str(cont))
            
            result_id_list.append(id_list)
            result_comment_list.append(comment_list)

            # 한번에 댓글이 20개씩 보이기 때문에 한 페이지씩 몽땅 댓글을 긁어 옵니다.
            if int(total_comm) <= ((page) * 20):
                break
            else:
                page += 1
    return (flatten(result_id_list), flatten(result_comment_list))
 
 

 
 
# 리스트 결과입니다.

#allCommetns = flatten(List)
allComments = collect_comments()

#ids = allComments[0]
#comments = allComments[1]
#print(allComments)
#to_csv_file(ids[:4580],comments[:4580])
# print(len(allComments[0]))
# print(len(allComments[1]))
to_csv_file(allComments[0][:4500],allComments[1][:4500])