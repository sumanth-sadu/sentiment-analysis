#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[70]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import os
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import re
# import pickle
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS


# In[71]:


# get_ipython().system("pip install bs4 # in case you don't have it installed")
# get_ipython().system('pip install emot')


# ## Read Data

# In[72]:


#to get the current working directory
directory = os.getcwd()
url = os.path.join(directory, "data.tsv")
df = pd.read_csv(url, sep='\t', header=0, on_bad_lines='skip')


# ## Keep Reviews and Ratings

# In[73]:


df = df[['review_body','star_rating']]


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[74]:


df = df[df['star_rating'].eq(1) | df['star_rating'].eq(2) | df['star_rating'].eq(3) | df['star_rating'].eq(4) | df['star_rating'].eq(5)]
df['class'] = df['star_rating'].apply(lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3)
df.head(2)


# In[75]:


# there are few nan values in the dataframe, also removing duplicate rows
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index()


# In[76]:


class1 = df[df['class']==1].sample(n=20000, random_state=42)
class2 = df[df['class']==2].sample(n=20000, random_state=42)
class3 = df[df['class']==3].sample(n=20000, random_state=42)
df = pd.concat([class1, class2, class3])


# In[77]:


# average length before data cleaning:
before_cleaning = (df['review_body'].str.len()).mean()
# print('Average length of the reviews before data cleaning :', (df['review_body'].str.len()).mean())


# # Data Cleaning
# 
# 

# In[78]:


def remove_urls(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return (text)

def remove_contractions(text) :
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
        expanded_text = ' '.join(expanded_words)
    return expanded_text


remove_non_english = lambda s: re.sub(r'[^a-zA-z]', ' ', s)
remove_spaces = lambda s: re.sub(' +',' ', s)


# In[79]:


def cleaning(text):
    #remove urls
    text = remove_urls(text)
    #remove html tags
    text = BeautifulSoup(text, "lxml").text
    #remove contractions 
    text = remove_contractions (text)
    #remove non-alphabetic chars 
    text = remove_non_english(text)
    #lowercase
    text = text.lower( )
    #remove extra spaces 
    text = remove_spaces(text)
    
    return text


# In[80]:


df['cleaned_text_reviews'] = list(map(cleaning, df.review_body))


# In[81]:


def convert_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text

df['cleaned_text_reviews'] = df['cleaned_text_reviews'].apply(lambda row: convert_emojis(str(row)))


# In[82]:


# average length after data cleaning:
after_cleaning = (df['cleaned_text_reviews'].str.len()).mean()
# print('Average length of the reviews after data cleaning :', (df['cleaned_text_reviews'].str.len()).mean())
print(before_cleaning,' , ', after_cleaning)

# # Pre-processing

# In[83]:


# average length before pre-processing:
before_preprocessing = (df['cleaned_text_reviews'].str.len()).mean()
# print('Average length of the reviews before pre-processing :', (df['cleaned_text_reviews'].str.len()).mean())


# In[84]:


df2 = df.copy()


# ## remove the stop words 

# In[85]:


from nltk.corpus import stopwords

to_remove = ['not']
new_stopwords = set(stopwords.words('english')).difference(to_remove)

df2['cleaned_text_reviews'] = df2['cleaned_text_reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in new_stopwords))


# ## perform lemmatization  

# In[86]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df2['cleaned_text_reviews'] = df2['cleaned_text_reviews'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[87]:


# average length after pre-processing:
after_preprocessing = (df2['cleaned_text_reviews'].str.len()).mean()
print(before_preprocessing, ' , ', after_preprocessing)
# print('Average length of the reviews after pre-processing :', (df2['cleaned_text_reviews'].str.len()).mean())


# # TF-IDF Feature Extraction

# In[88]:


from sklearn.model_selection import train_test_split
x_train,x_valid,y_t,y_v = train_test_split(df2['cleaned_text_reviews'],df2['class'],test_size=0.2, stratify = df2['class'], random_state = 21)


# In[89]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 3))
tfidf.fit(df2['cleaned_text_reviews'])


# In[90]:


x_t = tfidf.transform(x_train)
x_v = tfidf.transform(x_valid)


# # Perceptron

# In[91]:


from sklearn.linear_model import Perceptron
p = Perceptron()
p.fit(x_t, y_t)


# In[92]:


report = classification_report(y_v, p.predict(x_v), output_dict=True )


# In[93]:


# print('Perceptron:')
# print('Class 1: Precision - ', report['1']['precision'], ', Recall - ', report['1']['recall'], ', F1-score - ', report['1']['f1-score'])
# print('Class 2: Precision - ', report['2']['precision'], ', Recall - ', report['2']['recall'], ', F1-score - ', report['2']['f1-score'])
# print('Class 3: Precision - ', report['3']['precision'], ', Recall - ', report['3']['recall'], ', F1-score - ', report['3']['f1-score'])
# print('Average: Precision - ', precision_score(y_v, p.predict(x_v), average='micro'), ', Recall - ', recall_score(y_v, p.predict(x_v), average='micro'), ', F1-score - ', f1_score(y_v, p.predict(x_v), average='micro'))


print(report['1']['precision'], ' , ', report['1']['recall'], ' , ', report['1']['f1-score'])
print(report['2']['precision'], ' , ', report['2']['recall'], ' , ', report['2']['f1-score'])
print(report['3']['precision'], ' , ', report['3']['recall'], ' , ', report['3']['f1-score'])
print(precision_score(y_v, p.predict(x_v), average='micro'), ' , ', recall_score(y_v, p.predict(x_v), average='micro'), ' , ', f1_score(y_v, p.predict(x_v), average='micro'))

# # SVM

# In[94]:


from sklearn.svm import LinearSVC
sv = LinearSVC()
sv.fit(x_t, y_t)


# In[95]:


report_sv = classification_report(y_v, sv.predict(x_v), output_dict=True )


# In[96]:


# print('SVM:')
# print('Class 1: Precision - ', report_sv['1']['precision'], ', Recall - ', report_sv['1']['recall'], ', F1-score - ', report_sv['1']['f1-score'])
# print('Class 2: Precision - ', report_sv['2']['precision'], ', Recall - ', report_sv['2']['recall'], ', F1-score - ', report_sv['2']['f1-score'])
# print('Class 3: Precision - ', report_sv['3']['precision'], ', Recall - ', report_sv['3']['recall'], ', F1-score - ', report_sv['3']['f1-score'])
# print('Average: Precision - ', precision_score(y_v, sv.predict(x_v), average='micro'), ', Recall - ', recall_score(y_v, sv.predict(x_v), average='micro'), ', F1-score - ', f1_score(y_v, sv.predict(x_v), average='micro'))


print(report_sv['1']['precision'], ' , ', report_sv['1']['recall'], ' , ', report_sv['1']['f1-score'])
print(report_sv['2']['precision'], ' , ', report_sv['2']['recall'], ' , ', report_sv['2']['f1-score'])
print(report_sv['3']['precision'], ' , ', report_sv['3']['recall'], ' , ', report_sv['3']['f1-score'])
print(precision_score(y_v, sv.predict(x_v), average='micro'), ' , ', recall_score(y_v, sv.predict(x_v), average='micro'), ' , ', f1_score(y_v, sv.predict(x_v), average='micro'))


# # Logistic Regression

# In[97]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_t, y_t)


# In[98]:


report_lr = classification_report(y_v, lr.predict(x_v), output_dict=True )


# In[99]:


# print('Logistic Regression:')
# print('Class 1: Precision - ', report_lr['1']['precision'], ', Recall - ', report_lr['1']['recall'], ', F1-score - ', report_lr['1']['f1-score'])
# print('Class 2: Precision - ', report_lr['2']['precision'], ', Recall - ', report_lr['2']['recall'], ', F1-score - ', report_lr['2']['f1-score'])
# print('Class 3: Precision - ', report_lr['3']['precision'], ', Recall - ', report_lr['3']['recall'], ', F1-score - ', report_lr['3']['f1-score'])
# print('Average: Precision - ', precision_score(y_v, lr.predict(x_v), average='micro'), ', Recall - ', recall_score(y_v, lr.predict(x_v), average='micro'), ', F1-score - ', f1_score(y_v, lr.predict(x_v), average='micro'))


print(report_lr['1']['precision'], ' , ', report_lr['1']['recall'], ' , ', report_lr['1']['f1-score'])
print(report_lr['2']['precision'], ' , ', report_lr['2']['recall'], ' , ', report_lr['2']['f1-score'])
print(report_lr['3']['precision'], ' , ', report_lr['3']['recall'], ' , ', report_lr['3']['f1-score'])
print(precision_score(y_v, lr.predict(x_v), average='micro'), ' , ', recall_score(y_v, lr.predict(x_v), average='micro'), ' , ', f1_score(y_v, lr.predict(x_v), average='micro'))


# # Naive Bayes

# In[100]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_t, y_t)


# In[101]:


report_nb = classification_report(y_v, nb.predict(x_v), output_dict=True )


# In[102]:


# print('Naive Bayes:')
# print('Class 1: Precision - ', report_nb['1']['precision'], ', Recall - ', report_nb['1']['recall'], ', F1-score - ', report_nb['1']['f1-score'])
# print('Class 2: Precision - ', report_nb['2']['precision'], ', Recall - ', report_nb['2']['recall'], ', F1-score - ', report_nb['2']['f1-score'])
# print('Class 3: Precision - ', report_nb['3']['precision'], ', Recall - ', report_nb['3']['recall'], ', F1-score - ', report_nb['3']['f1-score'])
# print('Average: Precision - ', precision_score(y_v, nb.predict(x_v), average='micro'), ', Recall - ', recall_score(y_v, nb.predict(x_v), average='micro'), ', F1-score - ', f1_score(y_v, nb.predict(x_v), average='micro'))


print(report_nb['1']['precision'], ' , ', report_nb['1']['recall'], ' , ', report_nb['1']['f1-score'])
print(report_nb['2']['precision'], ' , ', report_nb['2']['recall'], ' , ', report_nb['2']['f1-score'])
print(report_nb['3']['precision'], ' , ', report_nb['3']['recall'], ' , ', report_nb['3']['f1-score'])
print(precision_score(y_v, nb.predict(x_v), average='micro'), ' , ', recall_score(y_v, nb.predict(x_v), average='micro'), ' , ', f1_score(y_v, nb.predict(x_v), average='micro'))
