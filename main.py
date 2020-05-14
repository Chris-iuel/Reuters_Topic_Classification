from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 


#- Load data
print("Loading data...")
df = pd.read_csv('data\\reuters_data_train.csv')
df_test = pd.read_csv('data\\reuters_data_test.csv')

#- Prepare Text vectorization
#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

'''
Include articles with no topics to create a more complete idf
then remove all nans
'''
print("Preparing data...")
#- Remove nans
df.dropna(subset=['Topics'], inplace=True)
df.reset_index(drop=True, inplace=True)

df_test.dropna(subset=['Topics'], inplace=True)
df_test.reset_index(drop=True, inplace=True)

vectorizer.fit(df['Text'])

#- Transform data to vector representation
vect_tf_idf_train = vectorizer.transform(df['Text']).todense()
vect_tf_idf_test = vectorizer.transform(df_test['Text']).todense()

print("Preparing model...")

def _create_data_loader(data, target,batch_size):
    from models import Valid_Dataset
    dataset = Valid_Dataset(data, target)

    data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=False)

    return data_loader

#- Setup model
import models
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

#Parameters
num_epochs = 50
batch_size = 64
learning_rate = 1e-3

#- Make model and loaders 
#model = models.Single_layer().cuda()
#model = models.Mini_hourglass().cuda()
model = models.Standard().cuda()
train_dataloader = _create_data_loader(vect_tf_idf_train, df['labels'],batch_size)
test_dataloader = _create_data_loader(vect_tf_idf_test, df_test['labels'],batch_size)

#- Find cross entropy weights
'''
We only have weights for classes present in test set.
so pad to reach 82
'''
loss_weights = df['labels'].value_counts(normalize=True)
loss_weights = 1-loss_weights.sort_index()
loss_weights = loss_weights.tolist()
# Padd to get all 82 topics
padding = [1]*(82-len(loss_weights))
loss_weights.extend(padding)
loss_weights = torch.tensor(loss_weights)

#- Setup 
criterion = torch.nn.CrossEntropyLoss(weight=loss_weights).cuda()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=1e-5
    ) # Use RAdams, Lookahead?

#- Run model
print("Starting model...")
for epoch in range(num_epochs):
    correct = 0
    test_loss = 0
    model.train()
    for idx, (inp, target) in tqdm(enumerate(train_dataloader)):
        output = model(inp)
        loss = criterion(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #- Log epoch
    print("-----------------------")
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))

    test_loss /= len(df['Text'])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(df['Text']),
        100. * correct / len(df['Text'])))

    #- Test model

    if (epoch +1) % 3 == 0:
        with torch.no_grad():
            estimates = np.zeros((82,82))
            model.eval()
            correct = 0
            test_loss = 0
            for idx, (inp, target) in tqdm(enumerate(test_dataloader)):
                output = model(inp)

                pred = output.argmax(dim=1, keepdim=True)
                for pd,trg in zip(pred,target):
                    estimates[trg,pd] += 1

                correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss += F.nll_loss(output, target, reduction='sum').item()

            print("##################################")
            print('Test [{}], loss:{:.4f}'
                .format(epoch, loss.data))

            test_loss /= len(df_test['Text'])

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(df_test['Text']),
                100. * correct / len(df_test['Text'])))
            print("##################################")

fig, ax = plt.subplots(constrained_layout=True)
normed_matrix = normalize(estimates, axis=1, norm='l1')
ax.matshow(normed_matrix)

ax.set_title('Prediction / Estimation')
ax.set_xlabel('Prediction')
ax.set_ylabel('Target')

print(df[['labels','Topics']].drop_duplicates().sort_values(by=['labels']).to_string())

plt.show()
