#import libraries
import requests 
from bs4 import BeautifulSoup
import csv
import pandas as pd
from itertools import zip_longest
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string
nltk.download('wordnet')
nltk.download('omw-1.4')
#import nltk to tokenize the predicted data to be shown
import nltk.data
nltk.download('punkt')
import joblib
#key words to check  if url is real url for privacy policy
#key words to validate the url as privacy policy url
key_words=['Privacy Policy','Terms and Conditions','Terms in Use','Terms & Conditions','Privacy Notice','privacy policy','Data Policy','Terms of Use']
#list to append the content to it
texts=[]

def remove_modal_content(soup):
    modal_contents = soup.find_all('div', {'class': 'modal-content'})
    for modal_content in modal_contents:
        try:
            modal_content.decompose()
        except:
            continue
    return soup
def remove_forms(soup):
    forms = soup.find_all('form')
    for form in forms:
        try:
            form.decompose()
        except:
            continue
    return soup
def remove_nav_bars(soup):
    nav_bars = soup.find_all('nav', {'class': 'navbar'})
    for nav_bar in nav_bars:
        try:
            nav_bar.decompose()
        except:
            continue
    return soup
def remove_footers(soup):
    footers = soup.find_all('footer')
    for footer in footers:
        try:
            footer.decompose()
        except:
            continue
    return soup
def remove_headers(soup):
    headers = soup.find_all('header')
    for header in headers:
        try:
            header.decompose()
        except:
            continue
    return soup
def remove_links(soup):
    links = soup.find_all('a', href=True)
    for link in links:
        try:
            link.decompose()
        except:
            continue
    return soup

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext
stop = set(stopwords.words("english"))

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)
#remove punctation
def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

string.punctuation
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def remove_tags(text):
    tags_list = ['<p>' ,'</p>' , '<p*>',
             '<ul>','</ul>',
             '<li>','</li>',
             '<br>',
             '<strong>','</strong>',
             '<span*>','</span>',
             '<a href*>','</a>',
             '<em>','</em>','→']
    for tag in tags_list:
        text = text.replace(tag, '')
    return text

def remove_links(soup):
    # Find all links in the soup
    links = soup.find_all('a', href=True)

    # Remove the links from the soup
    for link in links:
        if not '@' in link.get('href', ''):
            link.extract()
    return soup
#extract texts from the URL
def policy_scraping(URL):
    texts=[]
    result=requests.get(URL)
    src=result.content
    soup=BeautifulSoup(src,"lxml")
    soup = remove_modal_content(soup)
    soup = remove_forms(soup)
    soup = remove_nav_bars(soup)
    soup = remove_footers(soup)
    soup = remove_headers(soup)
    soup = remove_links(soup)
    text=soup.find_all(["p","li","th","td"])
    for i in range(len(text)):
      texts.append(text[i].text)
    return texts
#Validate that url is privacy policy url
def check_url(texts,URL):
    newdf=pd.DataFrame(texts)
    #check if the url content contain at least one of key words or not

    URL_index=[]
    for i in range(len(newdf)):
        if any(word in newdf[0][i]for word in key_words):
            URL_index.append(i)
            if len(URL_index)>0:
                break
    #check if the url is url for privacy policy or not
    url = URL.replace('/', ' ')
    url = url.replace('-', ' ')
    if any(keyword.lower() in url.lower() for keyword in key_words):
        check=True
    elif (len(URL_index) > 0):
        check=True
    else :
        check="Invalid URL, not privacy policy page "
    return check
#pre processing for text to be ready for prediction
def text_preprocessing(texts):
    df1=pd.DataFrame(texts)
    df1 = df1.rename(columns={0: 'text'})
    df1['paragraph']=df1.index
    df1["text"] = df1.text.map(remove_URL)
    df1["text"] = df1.text.map(cleanhtml)
    df1['text'] = df1['text'].apply(remove_tags)
    df1["text"] = df1.text.map(remove_stopwords)
    df1['text'] = df1.text.map(lemmatize_text)
    # rejoin the words to scentences again
    for i in range(len(df1)):
     df1.text[i] = ' '.join(df1.text[i])
    #remove punctations
    df1["text"] = df1.text.map(remove_punct)
    #remove the pargraphs that contain less than 5 words
    small_index=[]
    for i in range(len(df1)):
      if (len((df1['text'][i]).split())<5):
        small_index.append(i)
    df1=df1.drop(df1.index[small_index])
    #reset the index after droping step
    df1.reset_index(drop=True, inplace=True)
    #save the paragraph,list ,tables to another dataframe
    df2=df1['text']
    df2=pd.DataFrame(df2)
    return df2,small_index
clean_text = lambda x: x.replace('\n', '')

