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


def runall(sentences):
    from lamp.load_thesis_data import load_thesis_data
    data = load_thesis_data()
    classifier = pipeline('sentiment-analysis',data['bacteria_lamp_network'] )
    label = []
    for sentence in sentences:
        label.append(classifier(sentence))
    return label
def getparent(p = True):
    import os
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    parent = os.path.abspath(os.path.join(this_dir, os.pardir))
    grandparent = os.path.abspath(os.path.join(parent, os.pardir))
    if p ==True:
        return parent
    else:
        return grandparent


def lamp_pipeline(  genome,
                    abstracts = 'local',
                    email='None',
                    api_key='None',
                    outfolder='Here',
                    retmax=20,
                    wordcloud = True,
                    have_genome=True,
                    run_prokka=True,
                    html_ = False,
                    by_species=True):
    import os
    import seaborn as sns
    from shutil import copyfile
    from datetime import datetime
    from lamp.datamine import datamine
    from lamp.genome import get_blast_species, setup_dbs, prokka, blast_short, get_card_genes
    from lamp.sentence import clean, tokenize_sentences, species_sentences, word_cloud, word_stats
    print('LAMP Pipeline Starting\n\n')
    date = datetime.now()
    parent = getparent()
    if outfolder =='Here':
        try:
            os.mkdir('lamp_output')
            outfolder = os.getcwd()+'/lamp_output'
            os.mkdir(outfolder+'/graphs')
            os.mkdir(outfolder+'/imgs')
            os.mkdir(outfolder+'/data')
            for img in os.listdir(parent+'/imgs'):
                copyfile(parent+'/imgs/'+img, outfolder+'/imgs/'+img)
        except:
            outfolder = os.getcwd()+'/lamp_output'
    else:
        try:
            os.mkdir(outfolder+'lamp_output')
            outfolder = outfolder+'lamp_output'
            os.mkdir(outfolder+'/graphs')
            os.mkdir(outfolder+'/imgs')
            os.mkdir(outfolder+'/data')
            for img in os.listdir(parent+'/imgs'):
                copyfile(parent+'/imgs/'+img, outfolder+'/imgs/'+img)
        except:
            outfolder = os.getcwd()+'/lamp_output'
    if have_genome==True:
        species = get_blast_species(genome, outfolder+'/data/')
        get_card_genes(genome, outfolder+'/data/')
        if run_prokka ==True:
            prokka(genome,outfolder+'/data/')
    else:
        species=genome

    # # NLP Here
    if abstracts == 'local':
        # print('Using Local Abstracts\n\n')
        abstracts = getparent(p=False)+'/data/pubmed/'
        lst=[i for i in os.listdir(abstracts) if species.replace(' ','_') in i]
        mab=[''.join(open(abstracts+i,'r').readlines()) for i in lst]
        #all clean sentences
        fc = []
        #all labels
        fl=[]
        #the outcome of each paper
        ech = [['abstract','total','pos','neg','pct']]
        for a,b in zip(mab,lst):
            sentences = []
            if by_species ==True:
                sentences+=species_sentences(tokenize_sentences(a), species)
            else:
                sentences+=tokenize_sentences(a)

            try:
                clean_sentences = [clean(i) for i in sentences]
                clean_sentences=[i for i in clean_sentences if i !='']
                label = runall(clean_sentences)

                pos = [i for i in label if i[0]['label']=='POSITIVE']
                ech.append([b,len(clean_sentences),len(pos),len(clean_sentences)-len(pos),len(pos)/len(clean_sentences)*100])
                fc+=clean_sentences
                fl+=label
            except:
                x=0
        pos = [i for i in fl if i[0]['label']=='POSITIVE']
        adf = pd.DataFrame(ech[1:],columns = ech[0])

    else:
        x=0
        # print('Mining Genomes From %s'%abstracts)

    # barrnap and 16s
    if have_genome==False:
        genome = 'None'
    vals = ['a','b','c',1,2,3]
    r = {}
    r['genome']=genome
    r['have_genome']=have_genome
    r['query']=species
    r['date']=date
    r['email']=email
    r['pct'] = len(pos)/len(fc)*100
    r['positive_counts'] = len(pos)
    r['negative_counts'] = len(fc)-len(pos)
    r['number_sentences']=len(fc)
    r['number_abstracts']=len(lst)
    r['by_abstract']=adf
    r['clean_sentences']=clean_sentences
    if html_==True:
        html = (''.join(open(parent+'/templates/base.html').readlines()))
        for k,v in r.items():
            html = html.replace(str(k),str(v))
        newhtml = open(outfolder+'/click_me.html','w')
        newhtml.write(html)
        newhtml.close()
    sns.color_palette("tab10")
    label=species
    ax=sns.regplot(r['by_abstract']['pos'],r['by_abstract']['neg'],label=label)
    ax.legend(loc='best')
    # print('LAMP Pipeline Complete!\nI love LAMP')
    return r
