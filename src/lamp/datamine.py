#!/usr/bin/env python3
'''Just Make Biopython DataMine Work With Less Key Strokes'''
def datamine(species, email, api_key='None', retmax = 500, db = 'pmc'):
    import os
    from Bio import Entrez
    if api_key != 'None':
        Entrez.api_key = api_key
    Entrez.email = email
    abstracts = []
    handle = Entrez.esearch(db="pubmed", retmax = retmax,term=species+'[ORGN] and humans', idtype="acc")
    record = Entrez.read(handle)
    search_results = record['IdList']
    for i in search_results:
        fetch_handle = Entrez.efetch(
            db=db,
            rettype="gb",
            retmode="text",
            id=i,
        )
        data = fetch_handle.read()
        fetch_handle.close()
        abstracts.append(data)
    return abstracts

def mineall(species_list):
    abstracts = []
    for i in species_list:
        abstracts.append(datamine(i))
    return abstracts
