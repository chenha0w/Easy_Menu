import requests
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup
#from ediblepickle import checkpoint
from collections import defaultdict
import pandas as pd
import re
import os
import streamlit as st
import pickle


search_path = 'https://api.yelp.com/v3/businesses/search'
urlyelp="https://www.yelp.com/biz/"
urlyelp_photo="https://www.yelp.com/biz_photos/"
session=FuturesSession(max_workers=5)

@st.cache_resource
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm",disable = ['ner','lemmatizer','textcat'])
    
#@checkpoint(work_dir=r'/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, #kwargs:args[0]+'_at_'+args[1]+'.pkl')
#@checkpoint(work_dir=r'C:\Users\Daniel\PycharmProjects\asda_check\easy_menu\Easy_Menu\cache',key=lambda args, kwargs:args[0]+'_at_'+args[1]+'.pkl')
@st.cache_data(max_entries=1000,persist=True)
def search(restaurant_name, location='US', return_nums=10):
    api_key=os.getenv('YELP_API_KEY')
    headers = {'Authorization': f"Bearer {api_key}"}
    search_param={'term':restaurant_name, 'location':location, 'limit':return_nums}
    businesses=requests.get(search_path, headers=headers, params=search_param).json().get('businesses')
    if not businesses:
        raise ConnectionError("Current YELP API KEY hit the rate limit or expired. Try another day or contact develpers")
    return businesses

def get_rest_info(restaurant_name, location='US'):
    businesses=search(restaurant_name,location)
    name=businesses[0].get('name')
    alias=businesses[0].get('alias')
    review_num=businesses[0].get('review_count')
    address=', '.join(businesses[0].get('location').get('display_address'))
    category=[]
    for x in businesses[0].get('categories'):
        category.append(x['alias'])
    return name, address, category, alias, review_num

#@checkpoint(work_dir=r'/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, kwargs:args[0]+'-#'+str(args[1])+'.pkl')
#@checkpoint(work_dir=r'C:\Users\Daniel\PycharmProjects\asda_check\easy_menu\Easy_Menu\cache',key=lambda args, kwargs:args[0]+'-'+str(args[1])+'.pkl')
def get_page(alias,offset):
    urlrest = urlyelp+alias
    pagehtml = session.get(urlrest, params={'start':offset})
    return pagehtml.result()

def scrape_review(alias,offset,review_dict):
    tries=0
    urlrest = urlyelp+alias
    pagehtml = session.get(urlrest, params={'start':offset})
    soup = BeautifulSoup(pagehtml.result().text)  
    review_block = soup.find('section',attrs={'aria-label':"Recommended Reviews"})
    while not review_block:
        pagehtml = session.get(urlrest, params={'start':offset})
        soup = BeautifulSoup(pagehtml.result().text)
        review_block = soup.find('section',attrs={'aria-label':"Recommended Reviews"})
        tries+=1
        if tries>10:
            raise ConnectionError("Sorry we can't fetch the data. Please try other time.")
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
                raise ConnectionError("Sorry we can't fetch the data. Please try other time.")
        photo_t=photo.find('span',attrs={'class':"css-1enow5j"}).text
        n_photo=int(re.findall('[0-9]+',photo_t)[0])
        return n_photo

#@checkpoint(work_dir='r/Users/chenhaowu/Documents/pythoncode/dataincubator/cache',key=lambda args, kwargs:args[0]+'-#photo-'+str(args[1])+'.pkl')
#@checkpoint(work_dir=r'C:\Users\Daniel\PycharmProjects\asda_check\easy_menu\Easy_Menu\cache',key=lambda args, kwargs:args[0]+'-photo-'+str(args[1])+'.pkl')
def get_photo_page(alias,offset=0):
    urlrest=urlyelp_photo+alias
    pagehtml=session.get(urlrest, params={'start':offset})
    return pagehtml.result()

def scrape_photo(alias,offset,photo_dict):
    urlrest=urlyelp_photo+alias
    pagehtml=session.get(urlrest, params={'start':offset})
    soup=BeautifulSoup(pagehtml.result().text)
    imgblock=soup.find('div',attrs={'class':"media-landing_gallery photos"})
    while not imgblock:
        pagehtml = session.get(urlrest, params={'start':offset})
        soup = BeautifulSoup(pagehtml.result().text)
        imgblock=soup.find('div',attrs={'class':"media-landing_gallery photos"})
        tries+=1
        if tries>10:
            raise ConnectionError("Sorry we can't fetch the data. Please try other time.")
    for block in imgblock.find_all('img'):
        caption=re.findall('\.(.*)',block.get('alt'))
        if caption:
            photo_dict['caption'].append(caption[0].strip())
            photo_dict['imglink'].append(block.get('src'))

def rev_clean(reviews):
    #reviews being series
    clean_review=reviews.str.replace(r'(?<=[.,;!?])(?=[^\s.?!])', ' ',regex=True)
    rep_dict={"can't": 'can not', "won't": 'will not', "shan't": 'shall not', "n't": ' not', "'s": ' is', "'ve": ' have', "'ll": ' will', "'re": ' are', "'m": ' am'}
    for key,value in rep_dict.items():
        clean_review=clean_review.str.replace(key,value)
    return clean_review

def to_sents(review):
    #review is a str
    nlp = load_spacy()
    doc = nlp(review)
    sentences = []
    for sentence in doc.sents:
        sentences.append(sentence.text)
    return sentences


def export_df(alias,review_num):
    
    save_file=os.path.join('./cache', 'export_'+alias+'.pkl')
    if os.path.exists(save_file):
        with open(save_file,'rb') as f:
            return pickle.load(f)
            
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
    df_review['review']=df_review['review'].str.replace('\xa0', '')

    clean_review=rev_clean(df_review.review)
    df_review['sentences']=clean_review.apply(to_sents)
    
    with open(save_file,'wb') as f:
        pickle.dump((df_review,df_photo),f)
        
    return df_review,df_photo

