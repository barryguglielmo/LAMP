
def getparent():
    import os
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    parent = os.path.abspath(os.path.join(this_dir, os.pardir))
    return parent

def prokka(genome, outfolder):
    import os
    # print('running prokka: this step may take a few minutes')
    os.system('prokka %s --outdir %s --force'%(genome, outfolder))

def barrnap(genome, outfolder):
    from Bio import SeqIO
    import os
    # print('Running Barrnap to Pull 16S Seqs from Assembly\n\n')
    dashes = [j for j, x in enumerate(genome) if x == "/"]
    try:
        gname = genome[dashes[-1]+1:]
    except:
        gname = genome
    os.system('barrnap -in %s -outseq %s'%(genome,outfolder+gname+'.barrnap.fna'))

    with open(outfolder+gname+'.barrnap.fna','r' )as h:
        for r in SeqIO.parse(h,'fasta'):
            if '16S' in str(r.description).upper():
                seq = str(r.seq)
    try:
        out = open(outfolder+gname+'.16s.fna','w')
        out.write('>%s\n%s\n'%(gname+'.16s.fna',seq))
        out.close()
        os.system('rm %s'%(genome+'.fai'))
        return outfolder+gname+'.16s.fna'
    except:
        # print('No 16S Found\n\n')
        os.system('rm %s'%(genome+'.fai'))

def makeblastdb(file,out):
    import os
    cwd = os.getcwd()
    parent = getparent()
    os.chdir(parent+'/lamp_dbs')
    h = open(file,'r').readlines()
    h=[i.replace(' ','_') for i in h]
    new = open(file,'w')
    [new.write(i) for i in h]
    new.close()
    os.system('makeblastdb -dbtype nucl -in %s -out %s'%(file,out))
    os.chdir(cwd)
    # print('Making blastdb %s\n\n'%out)


def blast_short(seq_name, outfolder, db):
    import pandas as pd
    import os
    # print('Running BLAST %s\n\n'%db)
    cwd = os.getcwd()
    parent = getparent()
    os.chdir(parent+'/lamp_dbs')
    #seqname out folder
    dashes = [j for j, x in enumerate(seq_name) if x == "/"]
    try:
        nn= seq_name[dashes[-1]+1:]
    except:
        nn= seq_name
    os.system('blastn -db %s -query "%s" -out "%s.tsv" -outfmt 6'%(db, seq_name, outfolder+nn+'.'+db))
    df = pd.read_csv(outfolder+nn+'.'+db+'.tsv', header = None, sep = '\t')
    df.columns = ['qseqid','sseqid','pident','length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore']
    df.to_csv(outfolder+nn+'.'+db+'.tsv',sep='\t',index = False)
    os.chdir(cwd)
    return df

def get_blast_species(genome, outfolder, db = 'tlp'):
    import pandas as pd
    seq_name = barrnap(genome, outfolder)
    df = blast_short(seq_name, outfolder, db)
    df = df.loc[df.length>300]
    df = df.sort_values(by = 'pident', ascending = False).reset_index(drop=True)
    species=df.sseqid.iloc[0]
    dashes = [j for j, x in enumerate(species) if x == "_"]
    # print('Species Identified as:')
    # print(species[dashes[1]+1:dashes[2]]+' '+species[dashes[2]+1:dashes[3]])
    # print('\n')
    return species

def get_card_genes(genome, outfolder, db = 'card'):
    import pandas as pd
    df = blast_short(genome, outfolder, db)
    df = df.sort_values(by = 'pident', ascending = False).reset_index(drop=True)
    amr = []
    des = []
    for i in list(df.sseqid.values):
        dashes = [j for j, x in enumerate(i) if x == "|"]
        amr.append(i[dashes[4]+1:i[dashes[4]:].index('_')+dashes[4]])
        des.append(i[i[dashes[4]:].index('_')+dashes[4]+1:i[dashes[4]:].index('[')+dashes[4]-1])
    # clean card genes
    df['amr_gene']=amr
    df['amr_resistance']=des
    return df

def download_dbs(db ='tlp'):
    import os
    cwd = os.getcwd()
    parent = getparent()
    os.chdir(parent)
    try:
        os.mkdir('lamp_dbs')
        os.chdir('lamp_dbs')
    except:
        os.chdir('lamp_dbs')
    if db =='gtdb':
        try:
            os.system('wget https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/genomic_files_reps/gtdb_genomes_reps.tar.gz')
            os.system('gunzip gtdb_genomes_reps.tar.gz')
            # print('Finished GTDB')
            # print('Dbs downloaded at %s'%os.getcwd())
        except:
            x=0
            # print('Something Went Wrong')
    if db =='tlp':
        try:
            os.system('wget ftp://ftp.ncbi.nlm.nih.gov:21/refseq/TargetedLoci/Bacteria/bacteria.16SrRNA.fna.gz')
            os.system('gunzip bacteria.16SrRNA.fna.gz')
            # print('TLP Downloaded')
            # print('Dbs downloaded at %s'%os.getcwd())
        except:
            x=0
            # print('Something Went Wrong')
    if db =='card':
        from shutil import copyfile
        try:
            os.system('mkdir card')
            os.system('cd card')
            os.system('wget https://card.mcmaster.ca/latest/data')
            os.system('bzip2 -d card-data.tar.bz2')
            os.system('tar tar -xvf card-data.tar')
            copyfile('nucleotide_fasta_protein_homolog_model.fasta',parent+'/lamp_dbs/card.fna')
            os.system('rm -r %s/card'%(parent+'/lamp_dbs'))
            # print('Card Downloaded')
            # print('Dbs downloaded at %s'%os.getcwd())
        except:
            x=0
            # print('Something Went Wrong')
    os.chdir(cwd)
def setup_dbs():
    import os
    cwd = os.getcwd()
    parent = getparent()
    download_dbs(db='tlp')
    download_dbs(db='card')
    makeblastdb('bacteria.16SrRNA.fna', 'tlp')
    makeblastdb('card.fna','card')
    os.chdir(cwd)
