# Reuters_Topic_Classification

# Description
A small example of how i approach a text classification problem

# Build with
tqdm v 4.36.1
pandas v0.23.4
sklearn v0.20.3
numpy v1.15.4
BeautifulSoup v4.9.0
nltk v3.4.5

# How to
    Run extract_data.py
        Change the paths to point to the folder containing the data.

    Exploration.py 
        Gives some insights into the data. but is not needed to run the models.

    Run main.py
        Change loading paths.
        One of the three models can be run by changing the model loaded on line 64-66
        After the epochs has been completed, a plot is created. 
        Each row in the plot shows the target label of a topic 
        and the percentage of times it was miss classified as another topic.
        An overview of the labels and topics is printed out in the terminal.

# Modules

    ## Data Extraction

        Extracted data with Beautiful soup.
        Initially just merge the title and body together. 
        For future improvements we should emphasize the title more.

        Text cleaning
            Tokenize the text and remove stop words.


    ## Data exploration

        ### Nans   
            Alot of columns with no information
            Shows alot of missing topics
            Initial thoughs are keep the topics to fit the TF-IDF to give a broader sence of what words are truly unique,
            then remove the nans.
        
        ### Topics
            Not all topics in test are present in train. This makes it hard to assing topics to.
            Initial thoughs is correlate master list of topics with the fact book data to find similarities between topics and entities. 
            Then give words a weight for a certain class, even the ones not present in the training data.
            The distribution of topics are very skewed. 
            This indicates we could gain alot by assigning different weights to classes during training.
            This also indicates that generalisation on those topics will be hard, given the limmited training data.

        ### Tf-IDF inspection
            Fitting the vectoriser on the entire corpus yielded alot of irrelevant top scoring words. 
            This warrants a deeper look into what type of texts are present.
            A look into finding which words correlate with which classes would be a good way to reduce the amount of features
            removing articles without any topic, before fitting yiledes a lot more sensible word scores.

        ### Initial conclusions
            Remove nans before fitting TF-IDF vectorizer
            Keep all features to preserve the semantic meaning of the less represented topics.
            Use neural nets to deal with the resultihg high dimentionality data.

    ## Main
        Assign weights to the cross entropy loss based on the relative frequency of the topics.
        Setup a standard train / test environment for the model.


# Models
    Testing three simple Neural nets to estimate which architecture would be best suited for the task
    ## Single_layer
        A simple single layer NN used as a base line. 
        My initial thoughts are that the presence of specific words will be enough to estimate the topic.
 
    ## Standard
        A simple NN to test if more complex models yield more accurate Results

    ## Mini_hourglass
        The hour glass shape forces the network to create 
        a more efficient feature representation, and ignore irrelevant features. 


# Results
    85-88% Accuracy on all three models. This is inclusive to determine which model architecture is best.  
    I added in batch normalisation and drop out to prevent overfitting on the larger networks,
     as this was an apparent problem.
    A look at the prediction plot shows that most errors come from a couple of cases where the networks struggle.
    These cases are mostly unrelated eg target: money_supply pred: cpu.
    This points to some overfitting, and the wrong features being weighted to high. 
    I imagine this stems from some words unique to a few articles, but unrelated to their topics.
    This should be investigated for future improvements.


# Future improvements
    Correlating fact book data with topics and weighting words based on that.
    This correlation could also be used to identify the topics of some of the topic less articles,
    and increase the amount of available data.

    Use word correlation to classes instead of TF-IDF to score features.

    Cut irrelevant features, to mitigate overfitting.

    Investigate different network parameters.

    Use a domain trained word2vec representation and a convolutional neural net 
    to get a deeper insight into the data than a simple FC NN.

    Parameter tuning, little time was spend actualy tuning the models. 
    This could give some small gains, but more would be gained from preprocessing the data 

    Applying entity analysis and aditional NLP methods to preprocess the data before feeding it into the network 
    would probably yield the highest improvement in accuracy.

    Vectoriser doesn't include terms with a frequency lower than 5 by default.
    This means the topics with only one article, looses a few key features. 
    This could be mitigate by applying some domain knowledge and hand picking words from those texts.




    
