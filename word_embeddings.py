import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('word2vec_sample')
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

model.most_similar(positive=['university'], topn = 15)

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

medical_embeddings = lemmatized_words.difference(set(medical_terms))
print(medical_embeddings)

def determine_label(entity_labels, plaintiff_string):
    for term in medical_terms:        
        if term in plaintiff_string.casefold():
            return "MEDICAL"

    for term in medical_embeddings:        
        if term in plaintiff_string.casefold():
            return "MEDICAL_EMBED"
        
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