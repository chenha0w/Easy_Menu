import streamlit as st
import pandas as pd
import numpy as np
import getyelp2
import fuzzpair
import revcloud
import absaExtract
from PIL import Image

st.set_page_config(layout="wide", page_title="Easy Menu")
st.markdown('# Easy Menu :burrito: :sushi: :pizza: :curry:')
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
    df_photo=df_photo.drop(df_photo[df_photo.caption=='California burrito!'].index)
    df_review=absaExtract.to_sents(alias,df_review)
    df_extract=absaExtract.extract_rev(alias,df_review)
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
