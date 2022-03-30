import matplotlib.pyplot as plt
import pandas as pd
import nltk
import string
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
from nltk.corpus import stopwords
from nltk import FreqDist

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

stop_words = set(stopwords.words("english"))
new_stop_words = ["llc", "lp", "inc"]
stop_words.update(new_stop_words)

data = pd.read_csv('ojo_civ_cases_e.csv')
p = data['plaintiff'].to_frame()


medical_terms = ["health", "medical", "hospital", "clinic"]

named_entities = []
for index, row in p.iterrows():
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
        lbl = "NO_LABEL"
        if len(entity_labels) > 0:
            lbl = max(set(entity_labels), key = entity_labels.count)
            
            for term in medical_terms:        
                if term in words.casefold():
                    lbl = "MEDICAL"

        named_entities.append([lbl, words])
        
ner_frame = pd.DataFrame(named_entities, columns =['Entity', 'plaintiff'])
print(ner_frame.Entity.value_counts())
# plot = ner_frame.Entity.str.split(expand=True).stack().value_counts().plot.pie()
# plt.legend(ner_frame.Entity.value_counts().index.tolist(), loc="center")
# fig = plot.get_figure()
# fig.Position = [10, 10, 5500, 4000]
# fig.savefig('Oklahoma_Medical_Info.pdf', format='pdf', bbox_inches='tight')