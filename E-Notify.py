#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import pandas as pd
import tweepy
from tweepy import OAuthHandler
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import scipy
from scipy.sparse import csr_matrix
import numpy as np
from sklearn import tree
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn import linear_model


# In[2]:


consumer_key = 'bkK2DrDErBVyDDcH7k58Ky3AW'
consumer_secret = 'sHP8ryCUARGXfZMGicB2PdmiAC5V3uGdxtXi8foTUUe3DDo3r4'
access_token = '774855636383326208-1OUG5LiM3adHpY5INeYd9pvYai8JDvd'
access_token_secret = 'zvJ90xh3v1QOdCs6sdhahpem4fF6BHZHYmbEZN01SOv4X'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# In[3]:


keywords=["disaster",
"relief",
"earthquake",
"flood",
"hurricane",
"storm",
"tsunami",
"cyclone",
"landslide",
"avalanche",
"natural",
"volcano",
"fema",
"fire",
"police",
"shooting", 
"terrorism",
"nuclear", 
"attack", 
"catastrophe", 
"bombs", 
"calamity", 
"accident", 
"tragedy", 
"Epidemic" 
"evacuation",
"shaking", 
"aftershock", 
"disaster",
"evacuate", 
"evacuated", 
"tsunami", 
"flooding", 
"volcano", 
"eruption", 
"massive", 
"killed", 
"damage", 
"damaged", 
"destroyed", 
"destroy", 
"explosion"]


# In[4]:


month={
        'Jan' : 1,
        'Feb' : 2,
        'Mar' : 3,
        'Apr' : 4,
        'May' : 5,
        'Jun' : 6,
        'Jul' : 7,
        'Aug' : 8,
        'Sep' : 9, 
        'Oct' : 10,
        'Nov' : 11,
        'Dec' : 12
}


# In[5]:


tweetsCsv = pd.read_csv('tweets.csv', encoding = "ISO-8859-1", header=None, usecols=[2, 5])


# In[6]:


df = pd.DataFrame(tweetsCsv)    


# In[7]:


tweets=[]
for row in zip(df[2], df[5]):
    for keyword in keywords:
        if keyword in row[1]:
            split_date=row[0].split(" ")
            date=split_date[5]+"/"+str(month[split_date[1]])+"/"+split_date[2]
            clean_tweet= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[1]).split())
            tweet=(date, clean_tweet)
            tweets.append(tweet)


# In[8]:


len(set(tweets))


# In[9]:


tweets_from_api=[]
for keyword in keywords:
    tweets_from_api += api.search(q = keyword)


# In[10]:


listTuples=[]
for tweet in tweets_from_api: 
    clean_tweet= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text).split())
    split_date=tweet._json['created_at'].split(" ")
    date=split_date[5]+"/"+str(month[split_date[1]])+"/"+split_date[2]
    tweetTuple=(date, clean_tweet)
    listTuples+=tweetTuple


# In[11]:


listTuples


# In[12]:


allTweets = tweets+listTuples


# In[13]:


len(set(allTweets))


# In[14]:


allTweets


# In[15]:


indexed_tweets=[]
final_tweets=[]
i=1
for t in allTweets:
    today = datetime.date(2009, 4, 4)
    margin = datetime.timedelta(days = 100)
    if t is "" or t is None:
        continue
    sdate = t[0].split("/")
    if len(sdate)<3:
        continue
    if today - margin <= datetime.date(int(sdate[0]), int(sdate[1]), int(sdate[2])) <= today + margin:
        final_tweets.append(t)
        indexed_tweet=(i, t)
        i+=1
        indexed_tweets.append(indexed_tweet)


# In[16]:


len(indexed_tweets)


# In[17]:


classified_tweets = open('classifiedTweets.txt', encoding='utf8')


# In[18]:


classified_tweets= classified_tweets.read()


# In[19]:


len(classified_tweets)


# In[20]:


classified_tweets=classified_tweets.split("\n")


# In[21]:


classified_tweets[100]


# In[22]:


classified_tweets[105].split(",")[1]+"   "+ classified_tweets[105].split(",")[4]


# In[23]:


training=[]
training_label=[]
training_data=[]
classified_tweets[1].split(",")
for i in range(len(classified_tweets)-1):
    classified_tweet= classified_tweets[i].split(",")
    if len(classified_tweet)<=4:
        continue
    if classified_tweet[1] in ["Relevant", "Not Relevant"]:
        training_data.append(classified_tweet[4])
        label= 1 if classified_tweet[1]=="Relevant" else 0
        training_label.append(label)


# In[24]:


t = [final_tweet[1] for final_tweet in final_tweets]
x = list(training_data) + t


# In[25]:


test_x=np.asarray(training_data[:int(len(training_data)*0.20)])
training_data= np.asarray(training_data)
training_data.shape
test_x.shape

test_y=np.asarray(training_label[:int(len(training_label)*0.20)])
training_label= np.asarray(training_label)


# In[26]:


vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(x)


# In[27]:


X_test = vectors[int(len(training_data)*0.90):]
X_train = vectors[:int(len(training_data))]
Y_test = training_label[int(len(training_label)*0.90):]
Y_train = training_label[:X_train.shape[0]]


# In[28]:


def get_score(labels):
    return accuracy_score(Y_test, labels, normalize=True)


# In[29]:


logreg = linear_model.LogisticRegression()
def get_accuracy():
    countp=0
    countn=0
    global logreg
    logreg = logreg.fit(X_train, Y_train)
    predicted_labels = logreg.predict(X_test)
    i=0
    index_tweets=[]
    for p in predicted_labels:
        if p==1:
            countp+=1
            index_tweets.append(i)
        elif p==0:
            countn+=1
        i+=1
            
    relevant_tweets=[]
    for index in index_tweets:
        relevant_tweets.append(x[index])
    return relevant_tweets
#     return get_score(predicted_labels)


# In[30]:


from sklearn import linear_model
get_accuracy()


# In[31]:


rel_tweets= get_accuracy()
vectors = vectorizer.fit_transform(rel_tweets)
sum_freq = vectors.sum(axis=0)


# In[32]:


words_freq = [(word, sum_freq[0,idx]) for word, idx in vectorizer.vocabulary_.items()]


# In[33]:


words_freq = sorted(words_freq, key = lambda x:x[1], reverse= True)


# In[34]:


print(words_freq[:])


# In[35]:


final_rel_words=[]
for x,y in words_freq:
    if x in keywords:
        rel_word= [x,y]
        final_rel_words.append(rel_word)
print(final_rel_words)


# In[36]:


print(len(rel_tweets))


# In[37]:


top_tweets= rel_tweets[:5]
i=1
message_tweets="Top Tweets:"
for top_tweet in top_tweets:
    message_tweets+="\n"+str(i)+" " + top_tweet
    i+=1


# In[38]:


from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client


# In[39]:


app = Flask(__name__)
account_sid = 'AC927ba002034f4ef9fe90661ee1afce67'
auth_token = '0e549ab8532e6836eae8692d26eb5e3b'
client = Client(account_sid, auth_token)


# In[40]:


@app.route("/sms", methods=['GET', 'POST'])
def sms_ahoy_reply():
    #"""Respond to incoming messages with a friendly SMS."""
    # Start our response
    resp = MessagingResponse()
    #print(request.values.get('From'))
    # Add a message
    global message_tweets
    resp.message(message_tweets)
    tmp = ['+18578694005', '+18575448237', '+19789303476']
    for i in tmp:
        message = client.messages                 .create(
                     body=message_tweets,
                     from_='+18053605253',
                     to=i
                 )
    return str(resp)


# In[ ]:

if __name__ == "__main__":
    app.run()


# In[ ]:


message_tweets


# In[ ]:


quit


# In[ ]:




