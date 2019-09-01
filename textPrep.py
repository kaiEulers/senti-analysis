"""
This file contains the function textPrep() that performs lemmatisation and removes punctuations and stop words on the a collection of text
Created on Mon May 27 16:14:37 2019
@author: kaisoon
"""
def textPrep(docs, logging=True):
    """
    Returns a pandas series with each row containing white space seperated text that have been lemmatised with punctuations and stop words removed.
    If logging=True, function will track the progress of the preprocessing, as this can take a while if the collection of douments is wuite large.
    """
    import pandas as pd
    import string
    import nltk
    from nltk.corpus import stopwords
    import spacy
    nlp = spacy.load('en_core_web_sm')

    # --- Constants
    # Load stop words provided by Spacy
    sWords_spacy = set(spacy.lang.en.stop_words.STOP_WORDS)
    # Load stop words provided by NLTK
    sWords_nltk = set(stopwords.words('english'))
    # Load all punctuations
    punc = string.punctuation

    docs_prep = []
    cnt = 0
    # Loop thru documents in the doument collection
    for doc in docs:

    #    Log progress after every 100 completed processes
        cnt += 1
        if cnt%100 == 0 and logging:
            print(f"Processing {cnt} of {len(docs)}")

    #    Convert document into spacy object
        doc = nlp(doc)

#        Tokenise document and convert tokens to lowercase, remove unneccesary white space/s and punctuation/s
#        tokens = [tok.text.lower().strip() for tok in doc]

#        Tokenise document and convert tokens to its lemma in lowercase, remove unneccesary white space/s and punctuation/s
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

    #    Remove all stop words from tokens
#        tokens = [tok for tok in tokens if tok not in sWords_spacy]
        tokens = [tok for tok in tokens if tok not in sWords_nltk]

    #    Remove all punctuations from tokens
        tokens = [tok for tok in tokens if tok not in punc]

        # Contatenate tokens together, seperated by a white space
        doc_prep = ' '.join(tokens)
        docs_prep.append(doc_prep)

    print(f"\nPreprocessing Complete\n")
    return pd.Series(docs_prep)
