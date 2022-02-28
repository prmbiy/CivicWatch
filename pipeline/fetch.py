from os import write
from newspaper import Article
from datetime import datetime, time, timedelta
from pygooglenews import GoogleNews
from tqdm import tqdm
from csv import writer

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb 
import json
from pprint import pprint


class fetch():
  def __init__(self, country='India', lang='en'):

        self.country = country 
        self.lang = lang 
        self.gn = GoogleNews(lang=lang, country=country)

  def gather(self, query, time):
    last_hour = []
    pastHr = self.gn.search(query, when = time)
    last_hour.extend(pastHr['entries'])

    return last_hour

  def extract(self, query, fileName = 'last_hour', time = '1h'):
    self.last_hour = self.gather(query, time)

    with open(f'{fileName}.csv', 'a+') as inFile:
        for article in (self.last_hour):
            try:
                articleContent = Article(article.link)
                articleContent.download()
                articleContent.parse()

                data = [
                    article['title'],
                    article['published'],
                    articleContent.text,
                    article['link'],
                    article['source']['title'],
                    article['source']['href']
                ]

                writer_object = writer(inFile)
                writer_object.writerow(data)

            except Exception:
                continue 
    inFile.close()

    df = pd.read_csv(f'{fileName}.csv', names=['articleHead','publicationDate','articleBody','googleNews_link','sourceName','sourceLink'])
    return df    