import pandas as pd
import numpy as np
from collections import defaultdict
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

df_train = pd.read_csv('data\\reuters_data_train.csv')
df_test = pd.read_csv('data\\reuters_data_test.csv')


'''
Removing words can be shady here since there are few examples of some categories
'''
#- Info
print("\n====== Info ======")
print(df_train.info())
print(df_test.info())

#- Duplicate
print("\n====== Check for duplicates ======")
print(any(df_train.duplicated()))

#- Features
print("\n====== Number of words and unique words ======")
print("Words:", df_train['Text'].size, "Unique:", df_train['Text'].nunique())
print(df_train['Text'])

#- Labels
train_topics = pd.DataFrame(df_train['Topics'])
test_topics = pd.DataFrame(df_test['Topics'])

print("\n====== Unique Labels ======")
print(df_train['labels'].nunique())
print(df_test['labels'].nunique())

print("\n====== Train topic count ======")
print(df_train['Topics'].value_counts())

print("\n====== Test topic count ======")
print(df_test['Topics'].value_counts())

print("\n====== Exclusive classes ======")
exclusive = pd.DataFrame(pd.concat([train_topics,test_topics]).drop_duplicates(keep=False))
print("Exclusive")
print(exclusive)
print("Train exclusive")
print(train_topics.merge(exclusive, how = 'inner' ,indicator=False))
print("Test exclusive")
print(test_topics.merge(exclusive, how = 'inner' ,indicator=False))

print("\n====== Tf-IDF ======")

#- Remove nans
df_train.dropna(subset=['Topics'], inplace=True)
df_train.reset_index(drop=True, inplace=True)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df_train['Text']).todense()

default = np.zeros(vectors.shape[-1])
class_vectors = defaultdict(lambda: default.copy())

for topic, vec in tqdm(zip(df_train['Topics'],vectors)):
    class_vectors[topic] += np.asarray(vec)[0]


vocab = np.array(vectorizer.get_feature_names())
for topic,vec in class_vectors.items():
    top_5_idx = vec.argsort()[-5:][::-1]
    print(topic,vocab[top_5_idx])
