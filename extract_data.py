from bs4 import BeautifulSoup
import glob
from tqdm import tqdm
import pandas as pd

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenization(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words


#- Set paths!!
load_path = 'C:\\Users\\Christopher Iuel\\Documents\\Datasets\\reuters\\data\\'
save_path = "data\\"

topics = []
people = []
orgs = []
companies = []
text = []
data_split = []

#- Load up files
for filename in tqdm(sorted(glob.glob(load_path + '*.sgm'))):
    document = BeautifulSoup(open(filename), 'html.parser')
    all_articles = document.find_all('reuters')

    for article in all_articles:
        '''
        Uses Beautiful soup to extract the data from tags 
        '''
        
        #- Ternary operators to check for None, if its not, return content
        topics.append(article.topics.d.string if article.topics.d is not None else None)
        people.append(article.people.string if article.people is not None else None)
        orgs.append(article.orgs.string if article.orgs is not None else None)
        companies.append(article.companies.string if article.companies is not None else None)
        
        #- Add title to body to get more text
        txt = []
        txt = tokenization(article.title.string) if article.title is not None else [None]
        txt.extend(tokenization(article.body.string) if article.body is not None else [None])
        text.append(txt)

        data_split.append(article.get('lewissplit'))

#- Create df
df = pd.DataFrame(list(zip(topics,people,orgs,companies,text,data_split)), 
               columns =['Topics', 'People','Orgs','Companies','Text','Data_Split']) 
df.Topics = pd.Categorical(df.Topics)

#- labels = numerical version of topics
df['labels'] = df.Topics.cat.codes

#- Split data
df_train = df[df['Data_Split'] == 'TRAIN'].reset_index()
df_test = df[df['Data_Split'] == 'TEST'].reset_index()

#- Output df info
print(df.head())
print(df_train.head())
print(df_test.head())
print("Shape:",df.shape)

#- Save data
print("Saving data...")
df_train.to_csv(save_path + 'reuters_data_train.csv', index=True)
df_test.to_csv(save_path + 'reuters_data_test.csv', index=True)
print("Saved!")
