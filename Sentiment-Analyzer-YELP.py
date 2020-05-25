#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary packages
import requests
import json

import matplotlib.pyplot as plt
import numpy as np

import random
from pprint import pprint


# In[2]:


#Assigning the API Key to the Yelp_Token
YELP_TOKEN = "wrUx-8kY7vS4gfl4F-R2K4mG970aKsomVfoaURHiwZc3kHLAl4nEQ5svCAm8Zm0S6hd2N2n3RtuGaTV29d3FWhVvejMXRmNF6o42Hv0DUb5TmfR1esUi7YuhyD0FXnYx"


# In[3]:


#Creating a request object using get method
r = requests.get("https://api.yelp.com/v3/businesses/search?location=Toronto&limit=50", headers={"Authorization": "Bearer %s" % YELP_TOKEN})


# In[4]:


#Printing the status code, reason and content of the request 'r'
print(r.status_code, r.reason, r.content)


# In[5]:


#Setting the seed for consistant results
random.seed(1)
#Retrieving all and only url's from the request 'r'
url_labels = []
for business in r.json()['businesses']:
        url_labels.append((business['url']))


# In[6]:


#Prining the list url_labels and its length
print(url_labels)
len(url_labels)


# In[7]:


#importing BeautifulSoup
import requests
from bs4 import BeautifulSoup


# In[8]:


#Creating an empty list to later append reviews and ratings
review_labels = []
#For each url, we get the reviews and ratings under the script tag using soup
for url in url_labels:
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    all_review_data = json.loads(soup.find_all('script')[7].text)
    reviews = [each['description'] for each in all_review_data['review']]
    ratings = [each['reviewRating']['ratingValue'] for each in all_review_data['review']]
    review_labels.extend([[i,j] for i,j in zip(reviews,ratings)])


# In[9]:


#Printing the resulting appended list
review_labels


# In[10]:


#Writing the results to a json file
with open('data.json', 'w') as outfile:
    json.dump(review_labels, outfile)


# In[11]:


review_features = []

#Reading contents from the saved json file and then performing review split on space as well as checking if the ratings are greater than 3 for a particular review or not
with open('data.json') as json_file:
    full_file = json.load(json_file)
    review_features = [(x.split(' '), 'positive' if y > 3 else 'negative') for (x, y) in review_labels]


# In[12]:


#Displaying the results
review_features


# In[14]:


#Perforing sentiment analysis by splitting the results obtained from previous operation as training and test
from nltk.sentiment import SentimentAnalyzer
import nltk.sentiment.util
from nltk.classify import NaiveBayesClassifier

random.shuffle(review_features)
training_docs = review_features[:700]
test_docs = review_features[700:]

print("Training: %d, Testing: %d" % (len(training_docs), len(test_docs)))

sentim_analyzer = SentimentAnalyzer()


# In[15]:


all_words_neg = sentim_analyzer.all_words([nltk.sentiment.util.mark_negation(doc) for doc in training_docs])
all_words_neg


# In[16]:


unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigram_feats)


# In[17]:


training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(test_docs)


# In[18]:


#Calculating and displaying the required accracy, F-measurefor positive, negative and so on.
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))


# In[ ]:


#As can be seen, the classifier performs quite well on the data set with an accuracy of 82%. It precisiona nd F-measure for positive is also very high - 89%.

