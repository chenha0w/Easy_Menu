import streamlit as st
import pandas as pd
import numpy as np
import getyelp2
import fuzzpair
import revcloud

st.title("Easy Menu")
st.markdown("#")
result=st.empty()
with result.container():
    for n in range(1,20):
        st.markdown("#")

with st.sidebar:
    option = st.selectbox(
    'How would you like to input?',
    ('Input manually', 'Provide grubhub menu link', 'Take a picture of menu'),key='inputways')
    st.write("#")
    st.write("#")
    st.divider()

if option=='Input manually':
    with st.sidebar:
        restaurant=st.text_input("restaurant name", key="restaurant", placeholder="Sayulitas")
        location=st.text_input("location", key="location", placeholder="Mira Mesa")
    noshowdish=True
    
    if restaurant and location:
        name, address, category=getyelp2.get_rest_info(restaurant,location)
        wait="fetching data of {} at {} for you...".format(name,address)
        result.write(wait)
        df_review,df_photo=getyelp2.export_df(restaurant,location)
        done="Done! got data of {} at {}.Please type the dish name on the left.".format(name,address)
        result.write(done)
        noshowdish=False
    with st.sidebar:
        dish=st.text_input("Please enter the dish name:", key="dish", disabled=noshowdish, placeholder="california burrito")
    
    if dish:
        imglink=fuzzpair.select_photo(dish,df_photo)
        dishreview=fuzzpair.get_reviews(dish,df_review)
        wordcloud,wc_pos,wc_neg=revcloud.build2cloud(name,address,category,dish,dishreview)
        with result.container():
            if imglink:
                st.write("Don't like these photos?:thinking_face::thinking_face:  Refresh to see others:smile:")
                width=400 if len(imglink)==1 else 225/len(imglink)*3
                st.image(imglink,width=width)
            else:
                st.write("sorry we can't find any photo")
            st.pyplot(fig=wordcloud)
            st.subheader(dish)
            n_dishrev=dishreview.shape[0]
            rev_show,sort_word=revcloud.show_rev(wc_pos,wc_neg,dish,dishreview)
            with st.expander(f"Obtained {n_dishrev} reviews. Click to see some of them:"):
                if rev_show:
                    for i in range(0,len(rev_show)):
                        st.text_area('key word: '+sort_word[i],rev_show[i])
#                for x in dishreview.sample(n=min(7,n_dishrev)).review:
#                    sample_rev=fuzzpair.rev_print(dish,x)
#                    if sample_rev:
#                        st.text_area('',sample_rev,label_visibility='collapsed')  
                 
if option=='Provide grubhub menu link':        
    with st.sidebar:
        link=st.text_input("grubhub menu link", key="link", placeholder="https://...")
    if link:
        result.write('Sorry, this module is under development...')
        
if option=='Take a picture of menu':        
    with st.sidebar:
        uploaded_file = st.file_uploader("upload a menu photo")
    if uploaded_file:
        result.write('Sorry, this module is under development...')

