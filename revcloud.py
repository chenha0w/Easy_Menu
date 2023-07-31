from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import math

nlp = spacy.load("en_core_web_sm",disable = ['ner','lemmatizer','textcat'])

def common_stop(movein=set(),moveout=set()):
    keep={'empty','many','any','amount','full','enough','various','several','more','least','most','down','few','over','under'}.union(moveout)
    remove={'place','order','ordered','time','menu','food','menus','pretty'}.union(movein)
    stopwords=STOP_WORDS.union(STOPWORDS).difference(keep).union(remove)
    return stopwords

def unique_stop(name,address,category,dish):
    definestop=[]
    definestop.append(name.lower())
    definestop.extend(name.lower().split())
    definestop.extend(address.lower().split())
    definestop.extend([cat.lower() for cat in category])
    definestop.extend(dish.lower().split())
    definestop.append(dish)
    return definestop

def clean_tag(review):
    doc=nlp(review)
    cleaned_rev=[token.text for token in doc if token.tag_ in ['NN','JJ','JJR','NNP','RB']]
    return ' '.join(cleaned_rev)

def build2cloud(name,address,category,dish,dishreview):
    #clean reviews to keep only adv,adj,noun. Divide by ratings
    dishreview['clean_rev']=dishreview.review.apply(clean_tag)
    neg_revs=dishreview[dishreview['rating']<=3]
    pos_revs=dishreview[dishreview['rating']>3]
    ratio=math.sqrt(len(pos_revs)/len(neg_revs))
    neg_rev=neg_revs.clean_rev.str.cat(sep=' ')
    pos_rev=pos_revs.clean_rev.str.cat(sep=' ')

    #set stopwords
    stopwords=common_stop()
    definestop=unique_stop(name,address,category,dish)
    stopwords=stopwords.union(definestop)

    #get word frequency
    wordcloud_neg = WordCloud(stopwords=stopwords).generate(neg_rev.lower())
    wordcloud_pos = WordCloud(stopwords=stopwords).generate(pos_rev.lower())
    wordfreq_neg=wordcloud_neg.words_
    wordfreq_pos=wordcloud_pos.words_
    for word in wordfreq_neg.copy():
        for stop in definestop:
            if fuzz.ratio(word,stop)>80:
                del wordfreq_neg[word]
                break
    for word in wordfreq_pos.copy():
        for stop in stopwords:
            if fuzz.ratio(word,stop)>80:
                del wordfreq_pos[word]
                break
    
    #positive word cloud
    size=600
    x_pos, y_pos = np.ogrid[:size, :size]
    radius_pos=min(size/2,size/2*ratio)
    mask_pos = (x_pos - size/2) ** 2 + (y_pos - size/2) ** 2 > radius_pos ** 2
    mask_pos = 255 * mask_pos.astype(int)
    wc_pos = WordCloud(background_color = "white", mask = mask_pos,max_words=40).generate_from_frequencies(wordfreq_pos)

    #negaive word cloud
    x_neg, y_neg = np.ogrid[:size, :size]
    radius_neg=min(size/2,size/2/ratio)
    mask_neg = (x_neg - size/2) ** 2 + (y_neg - size/2) ** 2 > radius_neg ** 2
    mask_neg = 255 * mask_neg.astype(int)
    wc_neg = WordCloud(background_color = "white", mask = mask_neg, colormap='magma',max_words=40).generate_from_frequencies(wordfreq_neg)
    
    #plot word cloud
    fig,(ax_pos,ax_neg)=plt.subplots(1,2)
    ax_pos.imshow(wc_pos, interpolation='bilinear')
    ax_pos.axis("off")
    ax_pos.set_title('Pros',color='green')
    ax_neg.imshow(wc_neg, interpolation='bilinear')
    ax_neg.axis("off")
    ax_neg.set_title('Cons',color='red')
    
    return fig,wc_pos,wc_neg

def show_rev(wc_pos,wc_neg,dish,dishreview):
    rev_show=[]
    sort_word=[]
    neg_word=list(wc_neg.words_)[:5]
    pos_word=list(wc_pos.words_)[:5]
    words=set(pos_word+neg_word)
    for key,row in dishreview.sample(frac=1).iterrows():
        if not words or len(rev_show)>=7:
            break
        doc=nlp(row.review)
        prev=0
        for x in doc.sents:
            isbreak=0
            if fuzz.token_set_ratio(dish,doc[prev:x.end])>90 or fuzz.partial_ratio(dish.lower(),doc[prev:x.end].text.lower())>90:
                for word in words:
                    if word in x.text:
                        rev_show.append('...'+doc[max(0,prev-20):min(x.end+20,len(doc))].text+'...')
                        isbreak=1
                        sort_word.append(word)
                        words.remove(word)
                        break
            if isbreak:
                break 
            prev=x.start
    return rev_show,sort_word

def stopword_add(reviews,name,address,category):
    '''
    reviews is series
    '''
    control=reviews.str.cat(sep=' ')
    stopwords=set(STOPWORDS)
    definestop=[]
    definestop.append(name.lower())
    definestop.extend(name.lower().split())
    definestop.extend(address.lower().split())
    definestop.extend(['food','order','ordered'])
    definestop.extend(category)
    stopwords.update(definestop)
    wordcloudc = WordCloud(stopwords=stopwords).generate(control.lower())
    newstop=list(wordcloudc.words_.keys())[0:20]
    stopwords.update(newstop)
    return stopwords,definestop+newstop

def buildcloud(dish,stopwords,newadded,pairreview):
    '''
    pairreview is seires
    '''
    stopwords.update(dish.lower().split())
    stopwords.update(dish)
    catreview=pairreview.str.cat(sep=' ')
    wordcloud = WordCloud(stopwords=stopwords).generate(catreview.lower())
    wordfreq=wordcloud.words_
    for word in wordfreq.copy():
        for stop in set(newadded+dish.lower().split()):
            if fuzz.ratio(word,stop)>80:
                del wordfreq[word]
                break
    wc_update=WordCloud(width=200,height=200,background_color="white").generate_from_frequencies(wordfreq)     
    fig,ax=plt.subplots()
    ax.imshow(wc_update, interpolation='bilinear')
    ax.axis("off")
    return fig