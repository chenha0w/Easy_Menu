import requests
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup
from ediblepickle import checkpoint
from collections import defaultdict
import pandas as pd
import re
import os

search_path = 'https://api.yelp.com/v3/businesses/search'
urlyelp="https://www.yelp.com/biz/"
urlyelp_photo="https://www.yelp.com/biz_photos/"
session=FuturesSession(max_workers=5)

@checkpoint(work_dir=r'/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, kwargs:args[0]+'_at_'+args[1]+'.pkl')
def search(restaurant_name, location='US', return_nums=10):
    api_key=os.getenv('YELP_API_KEY')
    headers = {'Authorization': f"Bearer {api_key}"}
    search_param={'term':restaurant_name, 'location':location, 'limit':return_nums}
    businesses=requests.get(search_path, headers=headers, params=search_param).json().get('businesses')
    return businesses

def get_rest_info(restaurant_name, location='US'):
    businesses=search(restaurant_name,location)
    name=businesses[0].get('name')
    address=', '.join(businesses[0].get('location').get('display_address'))
    category=[]
    for x in businesses[0].get('categories'):
        category.append(x['alias'])
    return name, address, category

@checkpoint(work_dir=r'/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, kwargs:args[0]+'-'+str(args[1])+'.pkl')
def get_page(alias,offset):
    urlrest = urlyelp+alias
    pagehtml = session.get(urlrest, params={'start':offset})
    return pagehtml.result()

def scrape_review(alias,offset,review_dict):
    tries=0
    urlrest = urlyelp+alias
    pagehtml = session.get(urlrest, params={'start':offset})
    soup = BeautifulSoup(pagehtml.result().text,features='lxml')  
    review_block = soup.find('section',attrs={'aria-label':"Recommended Reviews"})
    while not review_block:
        pagehtml = session.get(urlrest, params={'start':offset})
        soup = BeautifulSoup(pagehtml.result().text,features='lxml')
        review_block = soup.find('section',attrs={'aria-label':"Recommended Reviews"})
        tries+=1
        if tries>10:
            raise RuntimeError("Sorry we can't fetch the data. Please try other time.")
    review_blocks = review_block.find_all('li', attrs={'class':'margin-b5__09f24__pTvws'})
    if len(review_block)>10:
        raise ValueError(f"got {len(review_block)} review blocks, more than maximal amount 10!")
    for block in review_blocks:
        review = block.find("span", attrs={'class':'raw__09f24__T4Ezm', 'lang':'en'})
        rating = block.find("div", attrs={'class':'five-stars__09f24__mBKym'})
        date = block.find("span", attrs={'class':'css-chan6m'})
        if bool(review) & bool(rating) & bool(date):  
            review_dict['review'].append(review.text.strip())
            review_dict['rating'].append(int(rating.get('aria-label')[0]))
            review_dict['date'].append(date.text)
    if offset==0:
        photo=soup.find('div',attrs={'class':"photo-header-buttons__09f24__UU4lV"})
        while not photo:
            pagehtml = session.get(urlrest, params={'start':offset})
            soup = BeautifulSoup(pagehtml.result().text,features='lxml')
            photo=soup.find('div',attrs={'class':"photo-header-buttons__09f24__UU4lV"})
            tries+=1
            if tries>10:
                raise RuntimeError("Sorry we can't fetch the data. Please try other time.")
        photo_t=photo.find('span',attrs={'class':"css-1enow5j"}).text
        n_photo=int(re.findall('[0-9]+',photo_t)[0])
        return n_photo

@checkpoint(work_dir='cache',key=lambda args, kwargs:args[0]+'-photo-'+str(args[1])+'.pkl')
def get_photo_page(alias,offset=0):
    urlrest=urlyelp_photo+alias
    pagehtml=session.get(urlrest, params={'start':offset})
    return pagehtml.result()

def scrape_photo(alias,offset,photo_dict):
    urlrest=urlyelp_photo+alias
    pagehtml=session.get(urlrest, params={'start':offset})
    soup=BeautifulSoup(pagehtml.result().text,features='lxml')
    imgblock=soup.find('div',attrs={'class':"media-landing_gallery photos"})
    while not imgblock:
        pagehtml = session.get(urlrest, params={'start':offset})
        soup = BeautifulSoup(pagehtml.result().text,features='lxml')
        imgblock=soup.find('div',attrs={'class':"media-landing_gallery photos"})
        tries+=1
        if tries>10:
            raise RuntimeError("Sorry we can't fetch the data. Please try other time.")
    for block in imgblock.find_all('img'):
        caption=re.findall('\.(.*)',block.get('alt'))
        if caption:
            photo_dict['caption'].append(caption[0].strip())
            photo_dict['imglink'].append(block.get('src'))

@checkpoint(work_dir=r'/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, kwargs:'export_'+args[0]+'_at_'+args[1]+'.pkl')
def export_df(name,location):    
    businesses=search(name,location)
    alias=businesses[0].get('alias')
    review_num=businesses[0].get('review_count')
    review_dict=defaultdict(list)
    photo_dict=defaultdict(list)
    for offset in range(0,review_num,10):
        if offset==0:
            n_photo=scrape_review(alias,offset,review_dict)
        else:
            scrape_review(alias,offset,review_dict)
    for offset in range(0,n_photo,30):
        scrape_photo(alias,offset,photo_dict)    
    df_review=pd.DataFrame(review_dict)
    df_photo=pd.DataFrame(photo_dict)
    rev_clean(df_review)
    return df_review,df_photo

def rev_clean(df_review):
    df_review['review']=df_review['review'].str.replace('\xa0', '')
