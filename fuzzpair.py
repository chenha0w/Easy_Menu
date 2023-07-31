from fuzzywuzzy import fuzz
import spacy
nlp = spacy.load("en_core_web_sm")

def fuzzselect(dishname,review):
    test=fuzz.token_set_ratio(dishname,review)>90 or fuzz.partial_ratio(dishname,review)>90
    if test:
        test=fuzz_check_rev(dishname,review)
    return test

def fuzz_check_rev(dishname,review):
    doc=nlp(review)
    prev=''
    test=False
    for x in doc.sents:
        if fuzz.token_set_ratio(dishname,prev+str(x))>90:
            test=True
        prev=str(x)
    #if fuzz.partial_ratio(dishname.lower(),review.lower())>90:
     #   test=True
    return test

def rev_print(dishname,review):
    doc=nlp(review)
    prev=0
    for x in doc.sents:
        if fuzz.token_set_ratio(dishname,doc[prev:x.end])>90 or fuzz.partial_ratio(dishname.lower(),doc[prev:x.end].text.lower())>90:
            return '...'+doc[max(0,prev-20):min(x.end+20,len(doc))].text+'...'
        prev=x.start
    return None

def get_reviews(dishname,df_review):
    reviews=df_review[df_review.review.apply(lambda x:fuzzselect(dishname,x))]   
    return reviews

def rank_photo(dishname,df_photo):
    pass
    

def get_photo(dishname,df_photo):
    candidates=df_photo[df_photo.caption.apply(lambda x:fuzzselect(dishname,x))]
    return candidates

def select_photo(dishname,df_photo):
    candidates=get_photo(dishname,df_photo)
    if not candidates.empty:
        choice=candidates.sample(n=min(3,candidates.shape[0]))
        return choice['imglink'].tolist()
    else:
        return None



