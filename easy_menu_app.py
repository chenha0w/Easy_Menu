import streamlit as st
import pandas as pd
import numpy as np
import getyelp2
import fuzzpair
import revcloud
import string
from collections import defaultdict
import os


@st.cache_resource
def load_absa():
    from pyabsa import AspectSentimentTripletExtraction as ASTE
    return ASTE.AspectSentimentTripletExtractor(checkpoint="english")



def extract_rev(alias,review):
    save_file=os.path.join('./cache', 'extract_aspect_of_'+alias+'.pkl')
    if os.path.exists(save_file):
        return pd.read_pickle(save_file)
            
    triplet_extractor = load_absa()
    dish_extract=defaultdict(list)
    for index,row in review.iterrows():
        for sentence in row.sentences:
            prediction=triplet_extractor.predict(sentence)
            extract=prediction.get('Triplets')
            if isinstance(extract,list):
                for i in range(0,len(extract)):
                    dish_extract['rev_index'].append(index)
                    dish_extract['sents'].append(prediction.get('sentence'))
                    aspect=extract[i].get('Aspect')
                    opinion=extract[i].get('Opinion')
                    dish_extract['aspect'].append(aspect)
                    dish_extract['opinion'].append(opinion)
                    dish_extract['phrase'].append(' '.join([opinion.strip(string.punctuation),aspect.strip(string.punctuation)]))
                    dish_extract['polarity'].append(extract[i].get('Polarity'))
    if not dish_extract:
        raise ImportError("pyabsa package crashed! Please close and reload the page. ")

    df_extract=pd.DataFrame(dish_extract)
    df_extract.to_pickle(save_file)
        
    return df_extract

st.set_page_config(layout="wide", page_title="Easy Menu")
st.markdown('# Easy Menu :burrito: :curry:')
st.markdown('## :wave: your ordering assistant :wave:')
st.markdown("## &emsp; _—learn more about a dish with pictures, wordclouds and reviews_")
st.markdown("##")
result=st.empty()
with result.container():
    for n in range(1,20):
        st.markdown("#")

#with st.sidebar:
#    option = st.selectbox(
#    'How would you like to input?',
#    ('Input manually', 'Provide grubhub menu link', 'Take a picture of menu'),key='inputways')
#    st.write("#")
#    st.write("#")
#    st.divider()

#if option=='Input manually':
with st.sidebar:
    st.title('Restaurant Name')
    restaurant=st.text_input("restaurant name", key="restaurant", placeholder="Sayulitas", label_visibility='collapsed')
    st.title('Location')
    location=st.text_input("location", key="location", placeholder="Mira Mesa", label_visibility='collapsed')
noshowdish=True

if restaurant and location:
    name, address, category, alias, review_num=getyelp2.get_rest_info(restaurant,location)
    wait="Fetching data of {} at {} for you...".format(name,address)
    result.info(wait)
    df_review,df_photo=getyelp2.export_df(alias, review_num)
    #df_review=absaExtract.to_sents(alias,df_review)
    df_extract=extract_rev(alias,df_review)
    if df_extract.empty:
        result.error('Oh no! pyabsa package crashed! Please close and reload the page.')
        st.stop()
    done="Done! got data of {} at {}.Please type the dish name on the left.".format(name,address)
    result.success(done)
    noshowdish=False
with st.sidebar:
    st.title('Please enter the dish name:')
    dish=st.text_input("Please enter the dish name:", key="dish", disabled=noshowdish, placeholder="california burrito", label_visibility='collapsed')

if dish:
    imglink=fuzzpair.select_photo(dish,df_photo)
    dishreview=fuzzpair.get_reviews(dish,df_review)
    wordcloud,wc_pos,wc_neg=revcloud.build2cloud_absa(name,address,category,dish,df_extract[df_extract['rev_index'].isin(dishreview.index)])
    with result.container():
        st.header(f':orange[{dish}]')
        if imglink:
            width=700 if len(imglink)==1 else 450/len(imglink)*3
            st.image(imglink,width=width)
            st.subheader(":orange[Don't like these photos?] :thinking_face::thinking_face:  :orange[Refresh to see others:smile:]")
        else:
            st.warning("Sorry we can't find any photo...")
        st.markdown("##")
        st.pyplot(fig=wordcloud)
        n_dishrev=dishreview.shape[0]
        #rev_show,sort_word=revcloud.show_rev(wc_pos,wc_neg,dish,dishreview)
        keyword,examples=revcloud.show_keyword(wc_pos,wc_neg,dish,dishreview,df_extract)
        st.markdown('### :orange[Curious about some words in wordclouds? Click below to check!]')
        with st.expander("Reviews with keywords:"):
            if keyword:
                for i in range(0,len(keyword)):
                    st.markdown(f'## **:blue[{keyword[i]}]**')
                    for j in range(0,len(examples[i])):
                        st.subheader(examples[i][j])
                #if rev_show:
                #    for i in range(0,len(rev_show)):
                #        st.text_area(f'key word: **:blue[{sort_word[i]}]**',rev_show[i])
#                for x in dishreview.sample(n=min(7,n_dishrev)).review:
#                    sample_rev=fuzzpair.rev_print(dish,x)
#                    if sample_rev:
#                        st.text_area('',sample_rev,label_visibility='collapsed') 


#if option=='Provide grubhub menu link':        
#    with st.sidebar:
#        link=st.text_input("grubhub menu link", key="link", placeholder="https://...")
#    if link:
#        result.write('Sorry, this module is under development...')
        
#if option=='Take a picture of menu':        
#    with st.sidebar:
#        uploaded_file = st.file_uploader("upload a menu photo")
#    if uploaded_file:
#        result.write('Sorry, this module is under development...')

