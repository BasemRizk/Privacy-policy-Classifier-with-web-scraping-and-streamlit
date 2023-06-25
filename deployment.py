#To run this script open the terminal and run (streamlit run deployment.py)
#This classifier is for english text only
#Some URLs to try:
#URL="https://www.edureka.co/privacy-policy"  (Valid privacy policy page)
#URL='https://policy.pinterest.com/en/privacy-policy' (valid privacy policy page)
#URL='https://www.coursera.org/about/privacy' (valid privacy policy page)
#URL='https://legal.yahoo.com/us/en/yahoo/privacy/index.html' (valid privacy policy page)
#URL='https://www.facebook.com/' (Invalid privacy policy page)

import streamlit as st
import joblib
import numpy as np
import nltk
from project_functions import *
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text_clf = joblib.load('privacy_clf.pkl')
target_names=['First Party Collection/Use','Third Party Sharing/Collection','User Choice/Control','User Access, Edit and  Deletion','Data Retention','Data Security','Policy Change','Do Not Track','International and Specific Audiences','Other']
#list the group names in an array
category_name=np.array(target_names)
# Set the title of the web page
st.title('Welcome to our Web Scraping Classifier')

# Add a text input box for the user to enter a URL
URL = st.text_input('Enter a URL')

# Add a button to submit the URL
if st.button('Submit'):
    # Process the URL
    # ...
    # Save the processed data to a variable
    processed_data = ...
    texts=policy_scraping(URL)
    newdf=pd.DataFrame(texts)
    check=check_url(texts,URL)
    if check==True:
        st.write('Data processed successfully!')
        df2,small_index=text_preprocessing(texts)
        #drop the text if the paragraph that has 0 or 1  words 
        newdf=newdf.drop(newdf.index[small_index])
        # Apply the lambda function to all elements in the DataFrame
        newdf = newdf.applymap(clean_text)
        newdf = newdf.drop_duplicates(subset=0)
        df2 = df2.drop_duplicates(subset='text')
        newdf.reset_index(drop=True, inplace=True)
        st.write('--------------------------------------------------------------')
        for i in range(len(df2)):
            pred=text_clf.predict(df2.iloc[i])
            
            data = newdf[0][i]
            st.markdown(f"### Paragraph Number: {i+1}")
            st.write('\n'.join(tokenizer.tokenize(data)))
            st.markdown("### Category :")
            st.write(category_name[pred][0])
            st.write("-----------------------------------------------------------")
    else:
        st.write(check)
