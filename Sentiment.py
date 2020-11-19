# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:53:41 2020

@author: Surya Prateek Soni
"""

################################################################################
# Working Directory
################################################################################
# Setting up the Working Directory 
import os
os.getcwd()
os.chdir("C:\\Users\\Surya Prateek Soni\\Desktop")

################################################################################
# Importing Libraries
################################################################################
import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
################################################################################
# Feature Exploration, Engineering and Cleaning
################################################################################
# Reading the Training and Testing dataset
train_data = pd.read_csv('twitter_sentiment_train.csv')
test_data = pd.read_csv('twitter_sentiment_test.csv')
# Combining both Train and Test for the operations where 
combine = [train_data, test_data]


# Checking the shape of Training and Testing datasets
# We have 27481 rows (tweets) in Training dataset and 3534 rows (tweets) in Test dataset
print(train_data.shape)
print(test_data.shape)


# Checking for Null and Non Null values 
train_data.info()
# We have just 1 Null Values in the Training set, we will just remove it
train_data.dropna(inplace=True)  # Removing the row with NAN
# Checking again if there are still any Nullvalues 
train_data.info()
# Confirmed - All rows have Non Null values
# Checking the same for Test Dataset also
test_data.info()
# Confirmed - All rows in Test dataset also have Non Null values




################################################################################
# EDA 
################################################################################

# Function to generate Random Colors 
# Input: number_of_colors - integer value indicating the number of colours which are going to be generated.
# Output: Color in the following format: ['#E86DA4'] .
def random_colours(number_of_colors):
    colors = []
    for i in range(number_of_colors):
        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    return colors

# Understanding the data 
train_data.head(5)
train_data.describe()

# Check the distribution of Tweets - Positve  Neutral and Negative in the Training dataset
temp = train_data.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
print(temp)
# Visualize it 
plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=train_data)

# Funnel-Chart 
fig = go.Figure(go.Funnelarea(text =temp.sentiment, values = temp.text, title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}))
fig.show()


# Function to calculate Jaccard Similarity 
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


#
results_jaccard=[]   # Empty List 
for ind,row in train_data.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text
    jaccard_score = jaccard(sentence1,sentence2)   # The "Text" and "Selected_Text" columns are given in as Inputs 
    results_jaccard.append([sentence1,sentence2,jaccard_score])  # The score calculated above is appended
# The variable "results_jaccard" will contain 3 things - Actual Text, Selected Text, and their Jaccard Score

# Creating a DataFrame called "Jaccard" so that we can merge it    
jaccard = pd.DataFrame(results_jaccard,columns=["text","selected_text","jaccard_score"])
train_data = train_data.merge(jaccard,how='outer')    # Merging it here 

# Calculating the number of words on both "text" and "Selected_Text"
train_data['Num_words_ST'] = train_data['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train_data['Num_word_text'] = train_data['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train_data['difference_in_words'] = train_data['Num_word_text'] - train_data['Num_words_ST'] #Difference in Number of words text and Selected Text


# Calculating the Distribution of Meta-Features
hist_data = [train_data['Num_words_ST'],train_data['Num_word_text']]
group_labels = ['Selected_Text', 'Text']
# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels,show_curve=False)
fig.update_layout(title_text='Distribution of Number Of words')
fig.update_layout(
    autosize=False,
    width=900,
    height=700,
    paper_bgcolor="LightSteelBlue")
fig.show()
    

# Distribution of Frequency VS Count of words 
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train_data['Num_words_ST'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')
p1=sns.kdeplot(train_data['Num_word_text'], shade=True, color="b")


# Differnce in Number of words and jaccard_scores across different Sentiments
# KDE Plot for Number of words across "Positive" and "Negative" sentiments 
# Blue is for Positive, Red for Negative 
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train_data[train_data['sentiment']=='positive']['difference_in_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')
p2=sns.kdeplot(train_data[train_data['sentiment']=='negative']['difference_in_words'], shade=True, color="r")
# KDE Plot for Neutral - The frequency of "0" is pretty high. 
plt.figure(figsize=(12,6))
sns.distplot(train_data[train_data['sentiment']=='neutral']['difference_in_words'],kde=False)


# KDE Plot for Jaccard Score across "Positive" and "Negative" sentiments 
# Blue is for Positive, Red for Negative 
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train_data[train_data['sentiment']=='positive']['jaccard_score'], shade=True, color="b").set_title('KDE of Jaccard Scores across different Sentiments')
p2=sns.kdeplot(train_data[train_data['sentiment']=='negative']['jaccard_score'], shade=True, color="r")
plt.legend(labels=['positive','negative'])
# KDE Plot for Jaccard Scorss for "Neutral" - The frequency of "0" is pretty high.  
plt.figure(figsize=(12,6))
sns.distplot(train_data[train_data['sentiment']=='neutral']['jaccard_score'],kde=False)
# Positive and negative tweets have high kurtosis and thus values are concentrated in two regions narrow and high density
# Neutral tweets have a low kurtosis value and their is bump in density near values of 1
# There is a peak in "Jaccard Similarity Score" for negative and positive plot around score of 1



# Finding the clusters of tweets where there is a high similarity between text and selected texts
# These clusters will help us find the "Selected_text" for "text"
# Lets start with tweets which have number of words lesss than 3 in text
k = train_data[train_data['Num_word_text']<=2]
k.groupby('sentiment').mean()['jaccard_score']
# As assumed, there is a "high similarity" between text and selected text
# Now let's look at each segment 
k[k['sentiment']=='positive']



###################################################################################################################
# DATA CLEANING
###################################################################################################################
# Function to clean the 2 columns 
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Cleaning the column called "Text"
train_data['text'] = train_data['text'].apply(lambda x:clean_text(x))
# Cleaning the column called "Selected Text"
train_data['selected_text'] = train_data['selected_text'].apply(lambda x:clean_text(x))
train_data.head()


# Most Common words in our Target-Selected Text
train_data['temp_list'] = train_data['selected_text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in train_data['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
print(temp)

# Visualizing the above Most Common Words in a Barplot
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# STOP WORDS
# Stop words need to be removed, and then again cleaning will be done. 
def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]  # Returning only Non-Stop words
train_data['temp_list'] = train_data['temp_list'].apply(lambda x:remove_stopword(x))   # Running the above function to remove the stop words

# Now counting the frequency of words.
top = Counter([item for sublist in train_data['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Purples')
print(temp)


# Plotting the tree of Most Common Words 
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()



# Most Common words in Text
train_data['temp_list1'] = train_data['text'].apply(lambda x:str(x).split()) #List of words in every row for text
train_data['temp_list1'] = train_data['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords

top = Counter([item for sublist in train_data['temp_list1'] for item in sublist])
temp = pd.DataFrame(top.most_common(25))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
print(temp)

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
# Most common words in Selected text and Text are almost the same



# Most common words Sentiments Wise
# Creating 3 different DataFrames for 3 different Sentiments
Positive_sent = train_data[train_data['sentiment']=='positive']
Negative_sent = train_data[train_data['sentiment']=='negative']
Neutral_sent  = train_data[train_data['sentiment']=='neutral']

# Now Deep Diving into every Sentiment 
# Most common positive words
top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')
print(temp_positive)
# Visualizing the Most common positive words on a Barplot
fig = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()


# Most common Negative words
top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative = temp_negative.iloc[1:,:]
temp_negative.columns = ['Common_words','count']
temp_negative.style.background_gradient(cmap='Reds')
print(temp_negative)
# Visualizing the Most common Negative words in a Tree Chart 
fig = px.treemap(temp_negative, path=['Common_words'], values='count',title='Tree Of Most Common Negative Words')
fig.show()


# Most common Neutral words
top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral = temp_neutral.loc[1:,:]
temp_neutral.columns = ['Common_words','count']
temp_neutral.style.background_gradient(cmap='Reds')
print(temp_neutral)
# Visualizing the Most common positive words on a Barplot
fig = px.bar(temp_neutral, x="count", y="Common_words", title='Most Commmon Neutral Words', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
# Visualizing the Most common Neutral words in a Tree Chart 
fig = px.treemap(temp_neutral, path=['Common_words'], values='count',title='Tree Of Most Common Neutral Words')
fig.show()
# words like get,go,dont,got,u,cant,lol,like are common in all three segments 




# Let's Look at Unique Words in each Segment
raw_text = [word for word_list in train_data['temp_list1'] for word in word_list]
def words_unique(sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

    '''
    allother = []
    for item in train_data[train_data.sentiment != sentiment]['temp_list1']:
        for word in item:
            allother .append(word)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in train_data[train_data.sentiment == sentiment]['temp_list1']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
    return Unique_words

# Unique Words for Tweets with Positve Sentiment 
Unique_Positive= words_unique('positive', 20, raw_text)
print("The top 20 unique words in Positive Tweets are:")
Unique_Positive.style.background_gradient(cmap='Greens')
print(Unique_Positive)
# Visualizing it in a Tree
fig = px.treemap(Unique_Positive, path=['words'], values='count',title='Tree Of Unique Positive Words')
fig.show()
# Visualizing in a Circular Pie Chart 
from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Positive Words')
plt.show()


# Unique Words for Tweets with Negative Sentiment 
Unique_Negative= words_unique('negative', 10, raw_text)
print("The top 10 unique words in Negative Tweets are:")
Unique_Negative.style.background_gradient(cmap='Reds')
print(Unique_Negative)
# Visualizing in a Circular Pie Chart 
from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.rcParams['text.color'] = 'black'
plt.pie(Unique_Negative['count'], labels=Unique_Negative.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Negative Words')
plt.show()



# Unique Words for Tweets with Neutral Sentiment 
Unique_Neutral= words_unique('neutral', 10, raw_text)
print("The top 10 unique words in Neutral Tweets are:")
Unique_Neutral.style.background_gradient(cmap='Oranges')
print(Unique_Neutral)
from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Neutral['count'], labels=Unique_Neutral.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Neutral Words')
plt.show()



# WordClouds
# Common Function for WordClouds
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}  # Adding more stop words 
    stopwords = stopwords.union(more_stopwords)   # Merging additional StopWords

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=400, 
                    height=200,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
d = '/kaggle/input/masks-for-wordclouds/'

# WORDCLOUD OF NEUTRAL TWEETS
pos_mask = np.array(Image.open(d+ 'twitter_mask.png'))
plot_wordcloud(Neutral_sent.text,mask=pos_mask,color='white',max_font_size=100,title_size=30,title="WordCloud of Neutral Tweets")

# WORDCLOUD OF NEUTRAL TWEETS


# WORDCLOUD OF NEUTRAL TWEETS





################################################################################################################
# Modeling 
################################################################################################################
df_train = pd.read_csv('twitter_sentiment_train.csv')
df_test = pd.read_csv('twitter_sentiment_test.csv')
#df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# Creating a new column- Number of words in column "Text"
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set

# Subsetting the DataFrame with only Rows where Number of Words are equal or greater than 3
df_train = df_train[df_train['Num_words_text']>=3]

def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = "C:\\Users\\Surya Prateek Soni\\Desktop"
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
        
        
        
# pass model = nlp if you want to train on top of existing model 

def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')        
    
    
    


def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path 


def get_training_data(sentiment):
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data


# Training models for Positive and Negative tweets
# Training models for Positive 
sentiment = 'positive'
trained_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
# For DEmo Purposes I have taken 3 iterations you can train the model as you want
train(trained_data, model_path, n_iter=3, model=None)    

# Training models for Negative
sentiment = 'negative'
trained_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)