import spacy
from ediblepickle import checkpoint
from pyabsa import AspectSentimentTripletExtraction as ASTE
import string
from collections import defaultdict
import pandas as pd

nlp = spacy.load("en_core_web_sm",disable = ['ner','lemmatizer','textcat'])


def rev_clean(review):
    review['review']=review['review'].str.replace(r'(?<=[.,;!?])(?=[^\s.?!])', ' ',regex=True)
    review['review']=review['review'].str.replace("can't", 'can not')
    review['review']=review['review'].str.replace("won't", 'will not')
    review['review']=review['review'].str.replace("shan't", 'shall not')
    review['review']=review['review'].str.replace("n't", ' not')
    review['review']=review['review'].str.replace("'s", ' is')
    review['review']=review['review'].str.replace("'ve", ' have')
    review['review']=review['review'].str.replace("'ll", ' will')
    review['review']=review['review'].str.replace("'re", ' are')
    review['review']=review['review'].str.replace("'m", ' am')

@checkpoint(work_dir=r'C:\Users\Daniel\PycharmProjects\asda_check\easy_menu\Easy_Menu\cache',key=lambda args, kwargs:'clean_rev_of_'+args[0]+'.pkl')
def to_sents(alias,review):
    rev_clean(review)
    review['sentences']=['']*len(review)
    for index,row in review.iterrows():   
        sentences=[]
        doc=nlp(row.review)
        for sentence in doc.sents:
            sentences.append(sentence.text)
        review.at[index,'sentences']=sentences
    return review

@checkpoint(work_dir=r'C:\Users\Daniel\PycharmProjects\asda_check\easy_menu\Easy_Menu\cache',key=lambda args, kwargs:'extract_aspect_of_'+args[0]+'.pkl')
def extract_rev(alias,review):
#    rev_clean(review)
#    review['sentences']=review.review.apply(to_sents)
    triplet_extractor = ASTE.AspectSentimentTripletExtractor(checkpoint="english")
    dish_extract=defaultdict(list)
    for index,row in review.iterrows():
        doc=nlp(row.review)
        for sentence in doc.sents:
            prediction=triplet_extractor.predict(sentence.text,ignore_error=False)
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
    return pd.DataFrame(dish_extract)