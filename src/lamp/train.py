#this may be all that is needed
from transformers import pipeline
import os
import pandas as pd
from transformers import *
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
NLP_SENTENCE_LABELS = [
    ('1','POSITIVE'),('0','NEGATIVE'),('2','NEUTRAL'),('-1','BAD SENTENCE'),]

def train(  trainset='bacteria_lamp.tsv',
            epochs = 10,
            outfolder = 'path/to/out/folder',
            add_drop = 'add_drop.tsv',
            train_size = 0.80,
            test_size = 0.10,
            valid_size = 0.10,
            shuffle=True,
            weight_decay=0.01,
            warmup_steps=500,
            learning_rate=1,
            seed = 0):

    # prep our sentences
    from lamp.sentence import clean
    print('\ntraining!\n')
    # read in our training csv
    df = pd.read_csv(trainset,sep = '\t').sample(frac=1).reset_index(drop=True)
    # follow the instructions from:
    # https://huggingface.co/transformers/custom_datasets.html
    train_labels = list(df.label.values)[:int(len(list(df.label.values))*train_size)]
    train_labels = [0 if x==1 else 1 for x in train_labels]
    test_labels  = list(df.label.values)[int(len(list(df.label.values))*test_size):int(len(list(df.label.values))*(test_size+train_size))]
    test_labels = [0 if x==1 else 1 for x in test_labels]
    val_labels   = list(df.label.values)[int(len(list(df.label.values))*(test_size+train_size)):]
    val_labels = [0 if x==1 else 1 for x in val_labels]
    #try swapping labels to train
    # 1 = Good; 0 = Bad

    texts = []
    for i in df.text:
        texts.append(clean(i, add_drop = add_drop))

    train_texts = texts[:int(len(texts)*(train_size))]
    test_texts  = texts[int(len(texts)*train_size):int(len(texts)*(test_size+train_size))]
    val_texts = texts[int(len(texts)*(test_size+train_size)):]

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings   = tokenizer(val_texts,   truncation=True, padding=True)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True)

    import torch
    class BioDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = BioDataset(train_encodings, train_labels)
    val_dataset = BioDataset(val_encodings, val_labels)
    test_dataset = BioDataset(test_encodings, test_labels)
    from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments,DistilBertForSequenceClassification
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        learning_rate=learning_rate,
    )
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.save_model('./results')
    tokenizer.save_pretrained("./results")
def guess(sentence, model='bl'):
    from transformers import pipeline
    from lamp.sentence import clean
    from lamp.load_thesis_data import load_thesis_data
    data = load_thesis_data()
    add_drop=data['add_drop']
    if model== 'bl':
        model = data['bacteria_lamp_network']
    print(sentence)
    classifier = pipeline('sentiment-analysis',model=model)
    return classifier((clean(sentence, add_drop = add_drop)))[0]['label']

def evaluate(eval_tsv, model, add_drop='add_drop.tsv'):
    from lamp.sentence import clean
    df = pd.read_csv(eval_tsv,sep = '\t').sample(frac=1).reset_index(drop=True)
    train_labels = list(df.label.values)
    train_conv = []
    for i in train_labels:
        if i ==0:
            train_conv.append('POSITIVE')
        elif i ==1:
            train_conv.append('NEGATIVE')
    classifier = pipeline('sentiment-analysis',model=model)
    # 1 = Good; 0 = Bad
    eval_labels = []
    for i in df.text:
        eval_labels.append(classifier((clean(i, add_drop = add_drop)))[0]['label'])
        # print(i)
        # print(eval_labels[-1])
    correct = 0
    ml = [['text','ann','human']]
    for k,v,t in zip(eval_labels,train_conv, list(df.text.values)):
        if k ==v:
            correct+=1
        else:
            ml.append([t,k,v])
    d = {}
    d['accuracy']=(correct/len(eval_labels))*100
    d['misslabeled']=pd.DataFrame(ml[1:], columns = ml[0])
    return d
def svm_train(dfs, dftags):
    pro = []
    con = []
    for df,j in zip(dfs,dftags):
        if j ==1:
            op = list(df['by_abstract'].pos.values)
            on = list(df['by_abstract'].neg.values)
            for x,y in zip(op,on):
                pro.append([x,y])
        elif j==2:
            op = list(df['by_abstract'].pos.values)
            on = list(df['by_abstract'].neg.values)
            for x,y in zip(op,on):
                con.append([x,y])
    pro_tag = [1 for i in pro]
    con_tag = [2 for i in con]
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array(pro+con)
    y = np.array(pro_tag+con_tag)
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    return clf.fit(X, y)
# clf = svm_train([lr,bl,cs,st],[1,1,2,2])
# clf.predict([[4,2]])
def svm_evaluate(svm, dfs, dftags):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    ox = []
    oy = []
    tx = []
    ty = []
    preds = []
    for df, t in zip(dfs, dftags):
        xs = []
        op = list(df['by_abstract'].pos.values)
        on = list(df['by_abstract'].neg.values)
        for x,y in zip(op,on):
            if t ==1:
                ox.append(x)
                oy.append(y)
            if t ==2:
                tx.append(x)
                ty.append(y)
            xs.append([x,y])
            pred = svm.predict(xs)
        preds.append([t,list(pred), np.average(pred)])
#     plt.scatter(ox,oy)
#     plt.scatter(tx,ty)
    label = ['Probiotics','Pathogens']
    import matplotlib.pyplot as plt
    plt.clf()
    ax=sns.regplot(ox,oy, label = 'Probiotics')
    ax=sns.regplot(tx,ty, label = 'Pathogens')
    ax.legend(loc=1)
    ax.plot(20)
    return pd.DataFrame(preds, columns = ['label','predictions','average'])
def kmeans_train(ds,
                n_init=10,
                gamma=1,
                degree=1,
                coef0=1,
                random_state=1,
                n_clusters=2
                ):
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    plt.clf()
    xs = []
    for d in ds:
        xs.append([d['positive_counts']/d['number_abstracts'],d['negative_counts']/d['number_abstracts']])
    X = np.array(xs)
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(n_clusters=n_clusters,
                                affinity='nearest_neighbors',
                                assign_labels='kmeans',
                                random_state=random_state,
                                gamma=gamma,
                                degree=degree,
                                n_init=n_init,
                                coef0=coef0)
    return model.fit_predict(X), xs
def graph_kmeans(xs, labels, title='KmeansTrain',ontop=False, pt=[0,0], label = 'none'):
    import matplotlib.pyplot as plt
    plt.clf()
    px = []
    py = []
    cx = []
    cy = []
    for j,k in zip(xs,labels):
        if k ==1:
            px.append(j[0])
            py.append(j[1])
        else:
            cx.append(j[0])
            cy.append(j[1])
    plt.scatter(px,py,cmap='PiYG', marker='s', label='Probiotics')
    plt.scatter(cx,cy,marker = 'o',label='Pathogens')
    if ontop==True:
        plt.scatter(pt[0],pt[1],c='r',s=100, marker='P', label=label)
    plt.title('Kmeans Prediction')
    plt.xlabel('Positive per Number Abstracts')
    plt.ylabel('Negative per Number Abstracts')
    plt.legend(loc=1)
    if ontop==True:
        plt.savefig(title+'%s.png'%label.replace(' ',''))
        # graph_kmeans(xs, labels, title=('RedrawnWith%s'%label))
    else:
        plt.savefig(title+'.png')



def graph_our_markers(ds,dtags):
    import matplotlib.pyplot as plt
    plt.clf()
    name = []
    gb=[]
    px = []
    py = []
    cx = []
    cy = []
    for d,t in zip(ds,dtags):
        name.append(d['query'])
        if t ==1:
            gb.append('probiotic')
            px.append(d['positive_counts']/d['number_abstracts'])
            py.append(d['negative_counts']/d['number_abstracts'])

        elif t ==0:
            gb.append('pathogen')
            cx.append(d['positive_counts']/d['number_abstracts'])
            cy.append(d['negative_counts']/d['number_abstracts'])
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(px,py,cmap='PiYG', marker='s', label='Probiotics')
    plt.scatter(cx,cy,marker = 'o',label='Pathogens')
    plt.legend(loc=1)
    plt.title('Our Labels')
    plt.xlabel('Positive per Number Abstracts')
    plt.ylabel('Negative per Number Abstracts')
    plt.savefig('KmeansOurLabels.png')
    df =  pd.DataFrame([name,gb,px+cx,py+cy]).transpose()
    df.columns = ['species','label','x','y']
    return df
def lamp_predict(query, ds, dtags, random_state=0, extra =False):
    from lamp.pipeline import lamp_pipeline
    import numpy as np
    from sklearn.svm import SVC
    from lamp.pipeline import lamp_pipeline
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    d = lamp_pipeline(query, have_genome=False)
    pt = [d['positive_counts']/d['number_abstracts'],d['negative_counts']/d['number_abstracts']]
    dtst=[d]+ds
    # originals
    labels,xs = kmeans_train(ds, random_state=random_state)
    # plot point on top of kmeans
    graph_kmeans(xs, labels,ontop=True, pt=pt, label = query)
    # if extra ==True:
    #     # new
    #     labels,xs = kmeans_train(dtst, random_state=random_state)
    #     # plot point on top of kmeans
    #     graph_kmeans(xs, labels,ontop=False)
    c=0
    for j,k in zip(dtags,labels):
        if j==k:
            c+=1
    print('Kmeans Percent Accuracy:%f'%(c/len(dtags)*100))
    if labels[0]==0:
        print('Kmeans Sentement: Probiotic')
    else:
        print('Kmeans Sentement: Pathogen')
    #svm
    plt.clf()
    pt = np.array([d['positive_counts']/d['number_abstracts'],d['negative_counts']/d['number_abstracts']])
    # originals
    X = np.array(xs)
    y = np.array(dtags)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    labels = clf.predict(X)
    c=0
    for j,k in zip(y,labels):
        if j==k:
            c+=1
    print('SVM Percent Accuracy:%f'%(c/len(y)*100))
    pred = clf.predict([pt])[0]
    if pred == 1:
        print('SVM Sentement: Probiotic')
    else:
        print('SVM Sentement: Pathogen')
    px = [i[0] for i,j in zip(X,labels) if j==1]
    py = [i[1] for i,j in zip(X,labels) if j==1]
    cx = [i[0] for i,j in zip(X,labels) if j==0]
    cy = [i[1] for i,j in zip(X,labels) if j==0]
    plt.clf()
    plt.scatter(px,py,marker='s', label='Probiotics')
    plt.scatter(cx,cy, label='Pathogens')
    plt.scatter(pt[0],pt[1], c='r',s=100,marker='P',label=query)
    plt.title('SVM Predictions')
    plt.legend(loc='best')
    plt.xlabel('Positive per Number Abstracts')
    plt.ylabel('Negative per Number Abstracts')
    plt.savefig('SVMPredictions.png')
