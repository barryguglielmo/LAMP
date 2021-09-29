#!/usr/bin/env python3
'''NLTK Tools'''
def clean(sentence, stemlem = 'lem', add_drop = 'add_drop.tsv',column = 'word', keep_all = False):
    import pandas as pd
    import nltk
    import os
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.lancaster import LancasterStemmer
    if stemlem == 'stem':
        stemlem = LancasterStemmer().stem
    elif stemlem == 'lem':
        stemlem = WordNetLemmatizer().lemmatize
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sentence.lower().replace('strain',''))
    token = [i for i in tokens if i.isdigit()==False]
    try:
        mystop = list(pd.read_csv(add_drop,sep='\t')[column].values)
        stop_words = stopwords.words('english')+mystop
    except:
        stop_words = stopwords.words('english')
        # print('no additional stop words added')
    stop_words = [i.lower() for i in stop_words]
    token = [i.lower() for i in token if i not in stop_words]
    #drop two letter words
    token = [i for i in token if len(i)>2]
    token = [stemlem(i) for i in token]
    if len(token)>3:
        sentence = ' '.join(token)
        return sentence
    else:
        return ''
def word_stats(sentences, mymin =5):
    counts = {}
    for sentence in sentences:
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        token = tokenizer.tokenize(sentence)
        for i in token:
            if i in counts.keys():
                counts[i]+=1
            else:
                counts[i]=1
    if mymin<0:
        counts = {key:val for key, val in counts.items() if val < abs(mymin)}
    else:
        counts = {key:val for key, val in counts.items() if val > mymin}
    return counts

def word_cloud(word_stats,cmap = 'gray', font_path = 'None'):
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import numpy as np

    if cmap == 'gray':
        greys = cm.get_cmap('Greys', 256)
        newcolors = greys(np.linspace(0, 1, 69))
        black = np.array([1/256, 1/256, 1/256, 1])
        newcolors[:25, :] = black
        cmap = ListedColormap(newcolors)
    words = []
    for k,v in word_stats.items():
        for i in range(0,v):
            words.append(k)
    words = [i for i in words if type(i) is not float]
    if font_path =='boom':
        font_path = '/home/boom/anaconda3/lib/fonts/DejaVuMathTeXGyre.ttf'
        wordcloud = WordCloud(width = 800, height = 800,
                        colormap=cmap,
                        background_color ='white',
                        font_path=font_path,
                        min_font_size = 10,collocations=False).generate(' '.join(words))

    elif font_path == 'None':
        wordcloud = WordCloud(width = 800, height = 800,
                        colormap=cmap,
                        background_color ='white',
                        min_font_size = 10,collocations=False).generate(' '.join(words))


        # plot the WordCloud image
    plt.figure(figsize = (4, 4), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    return plt

import os, re, nltk
def tokenize_sentences(text):
    import nltk
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    text = text.replace('\n',' ')
    sents = sent_detector.tokenize(text)
    return sents

def species_sentences(sentences, species, ispecies=True, include_genus=True):
    if ispecies==True:
        species = [species[:species.index(' ')],species,'%s. %s'%(species[0],species[species.index(' ')+1:])]
    else:
        species =[species, species, species]
    species_sentence = []
    for sentence in sentences:
        if include_genus==True:
            if species[0] in sentence or species[1] in sentence or species[2] in sentence:
                species_sentence.append(sentence)
        else:
            if species[1] in sentence or species[2] in sentence:
                species_sentence.append(sentence)
    return species_sentence
