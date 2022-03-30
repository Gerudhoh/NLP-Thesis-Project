import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('word2vec_sample')
import gensim
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
glove_model = api.load("glove-wiki-gigaword-50")

glove_model.most_similar("university")
model.most_similar(positive=['university'], topn = 15)

# Word2Vec

medical_terms = ["health", "medical", "hospital", "clinic"]
medical_embeddings = set()
for term in medical_terms:
    similars = model.most_similar(positive=[term], topn = 10)
    for wordembedding in similars:
        medical_embeddings.add(wordembedding[0].lower())
        
print(medical_embeddings)

lemmatizer = WordNetLemmatizer()
lemmatized_words = set()
for embedding in medical_embeddings:
    lemmatized = lemmatizer.lemmatize(embedding)
    lemmatized_words.add(lemmatized)

medical_embeddings_w2v = lemmatized_words.difference(set(medical_terms))
print(medical_embeddings_w2v)

# GLoVe
medical_terms = ["health", "medical", "hospital", "clinic"]
medical_embeddings_glove = set()
for term in medical_terms:
    similars = glove_model.most_similar(positive=[term], topn = 11)
    for wordembedding in similars:
        medical_embeddings_glove.add(wordembedding[0].lower())
        
print(medical_embeddings_glove)

lemmatizer = WordNetLemmatizer()
lemmatized_words = set()
for embedding in medical_embeddings_glove:
    lemmatized = lemmatizer.lemmatize(embedding)
    lemmatized_words.add(lemmatized)

medical_embeddings_glove = lemmatized_words.difference(set(medical_terms))
print(medical_embeddings_glove)

# Combine The Two
# medical_embeddings_w2v.update(medical_embeddings_glove)

# Sanitize
omit_words = ["poor", "care", "nearby", "education", "study", "public", "camp", "nutrition", "environmental", "hygiene", "veterinary"]
medical_embeddings_w2v = medical_embeddings_w2v.difference(set(omit_words))
medical_embeddings_glove = medical_embeddings_glove.difference(set(omit_words))

medical_embeddings_both = medical_embeddings_glove.intersection(medical_embeddings_w2v)
medical_embeddings_both = medical_embeddings_both.difference(set(omit_words))
print(medical_embeddings_both)

def determine_label(entity_labels, plaintiff_string):
    for term in medical_terms:        
        if term in plaintiff_string.casefold():
            return "MEDICAL"

    for term in medical_embeddings_both:    
        if term in plaintiff_string.casefold():
            return "MEDICAL_EMBED_BOTH"
        
    for term in medical_embeddings_w2v:        
        if term in plaintiff_string.casefold():
            return "MEDICAL_EMBED_W2V"
        
    for term in medical_embeddings_glove:   
        if term in plaintiff_string.casefold():
            return "MEDICAL_EMBED_GLOVE"
        
    if len(entity_labels) > 0:
        return max(set(entity_labels), key = entity_labels.count)
        
    return "NO_LABEL"

named_entities = []
for index, row in data.iterrows():
    words = row['plaintiff']
    entity_labels = []
    if(pd.notna(words)): 
        word_token = nltk.word_tokenize(words)
        POS_tags = nltk.pos_tag(word_token)
        ne = nltk.ne_chunk(POS_tags)
        for node in ne:
            if isinstance(node, nltk.Tree):
                if(node.label() != "GPE"):
                    entity_labels.append(node.label())
                    
        lbl = determine_label(entity_labels, words)
        named_entities.append([lbl, words])
        
ner_frame = pd.DataFrame(named_entities, columns =['Entity', 'plaintiff'])
ner_frame.Entity.value_counts()