#!/usr/bin/env python
# coding: utf-8
from datetime import datetime

import requests  # 发起网络请求
from bs4 import BeautifulSoup  # 解析HTML文本
import pandas as pd  # 处理数据
import os
import time  # 处理时间戳
import json  # 用来解析json文本

from tqdm import tqdm

'''
用于发起网络请求
url : Request Url
kw  : Keyword
page: Page number
'''


def fetchUrl(url, kw, page):
    # 请求头
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Connection": "keep-alive",
        "Cookie": "__jsluid_h=feed647f64ed868f713978e151ffee30; sso_c=0; sfr=1",
        "Host": "search.people.cn",
        "Origin": "http://search.people.cn",
        "Referer": "http://search.people.cn/s/?keyword=%E6%96%87%E5%8C%96&st=0&_=1640775018893",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62"
    }
    # 请求参数
    payloads = {
        "endTime": 0,
        "hasContent": True,
        "hasTitle": True,
        "isFuzzy": True,
        "key": kw,
        "limit": 10,
        "page": page,
        "sortType": 2,
        "startTime": 0,
        "type": 0,
    }

    # 发起 post 请求
    r = requests.post(url, headers=headers, data=json.dumps(payloads))
    return r.json()


def parseJson(jsonObj):
    # 解析数据
    records = jsonObj["data"]["records"]
    for item in records:
        # 这里示例解析了几条，其他数据项如末尾所示，有需要自行解析
        pid = item["id"]
        originalName = item["originalName"]
        belongsName = item["belongsName"]
        content = BeautifulSoup(item["content"], "html.parser").text
        displayTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["displayTime"] / 1000))
        subtitle = item["subtitle"]
        title = BeautifulSoup(item["title"], "html.parser").text
        url = item["url"]

        yield [[pid, title, subtitle, displayTime, originalName, belongsName, content, url]]


'''
用于将数据保存成 csv 格式的文件（以追加的模式）
path   : 保存的路径，若文件夹不存在，则自动创建
filename: 保存的文件名
data   : 保存的数据内容
'''


def saveFile(path, filename, data):
    # 如果路径不存在，就创建路径
    if not os.path.exists(path):
        os.makedirs(path)
    # 保存数据
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path + filename + ".csv", encoding='utf_8_sig', mode='a', index=False, sep=',', header=False)


if __name__ == "__main__":
    # 起始页，终止页，关键词设置
    start = 1
    end = 200
    kw_list = ["文化","经济","科技","农业","乡村","城镇","政治","历史","青年","人民","人民日报","十九届六中全会","进博会","抗疫","脱贫攻坚","数字经济","中国",
               "百年","疫苗","碳中和","反垄断","三胎","人口","生育","光刻机技术","5G","东数西算","光伏","新能源","芯片","反诈","饭圈","流量","税","退休","国际",
               "外交","美国","俄罗斯","气候","环境","环保","大小周","加班","躺平","创新","美丽中国","乡村振兴","品牌","主旋律","体育","运动","行业","责任","躺平",
               "旅游","冬奥","时代","生态","生活","知识","智慧","初心","工人","主义","鲁迅","贸易","金融","资本","国家","居民","数据","信息","安全","军事","边防",
               "法制","法治","年轻人","生活","奋斗","宗旨","发展","区域","城市化","财政","年龄","养老","社会","腐败","反腐","教育","菜","粮食","医疗","健康","市场",
               "时代","鲁迅","建党","铭记","不忘","纪念","工业","科学","交通","人工智能","淄博","全国统一","德","企业","中国式","外交","制度","行业","文化","孔乙己"]
    # 保存表头行
    headline = [["文章id", "标题", "副标题", "发表时间", "来源", "版面", "摘要", "链接"]]
    for kw in kw_list:
        saveFile("data/", "daily_" + kw, headline)
        # 爬取数据
        bar = tqdm(range(start, end + 1),
                   desc=f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word tokenisation",
                   total=end + 1,
                   ncols=150)
        for page in bar:
            url = "http://search.people.cn/search-platform/front/search"
            html = fetchUrl(url, kw, page)
            for data in parseJson(html):
                if datetime.strptime(data[0][3], '%Y-%m-%d %H:%M:%S').year < 2021:
                    continue
                saveFile("data/", kw, data)
            print("第{}页爬取完成".format(page))
            time.sleep(2)

    # 爬虫完成提示信息
    print("人民网爬虫执行完毕。数据已保存至以下路径中，请查看：")
    print(os.getcwd(), "\\data")
