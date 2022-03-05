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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

stop_words = set(stopwords.words("english"))
new_stop_words = ["llc"]
stop_words.append(new_stop_words)

data = pd.read_csv('Maricopa_Pew.csv', dtype={"FileDate": "string", "SubCategory": "string", "Immediate": "string", "Court": "string", "CaseStatus": "string", "CaseStatusDate": "string", "JudgmentFor": "string", "Zip": "string", "WritDate": "string", "PltLegalRep": "string", "DefLegalRep": "string", "ClaimAmount": "string", "BenchTrial": "string", "Plt2Name": "string", "year": int, "JudgmentAmount": "string", "Plt1Name": "string", "FilePurged": int})

p = data['Plt1Name'].to_frame()

p.Plt1Name.value_counts()

word_count = p.Plt1Name.str.split(expand=True).stack().value_counts()

# for index, row in p.iterrows():
#     words = row['Plt1Name']
#     if(pd.notna(words)): 
#         word_token = nltk.word_tokenize(words)
#         print(list(nltk.bigrams(word_token)))

# for index, row in p.iterrows():
#     words = row['Plt1Name']
#     if(pd.notna(words)): 
#         word_token = nltk.word_tokenize(words)
#         POS_tags = nltk.pos_tag(word_token)
#         NameEn = nltk.ne_chunk(POS_tags)
#         print(NameEn)


lower_data = p.Plt1Name.str.lower().to_frame()
word_count = lower_data.Plt1Name.str.split(expand=True).stack().value_counts()
print(word_count)

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for index, row in lower_data.iterrows():
    words = row['Plt1Name']
    if(pd.notna(words)): 
        lemmatizing_words = nltk.word_tokenize(words)
        new_lemmatized_words = [lemmatizer.lemmatize(word) for word in lemmatizing_words if word not in stop_words and word not in string.punctuation]
        lemmatized_words.append(' '.join(new_lemmatized_words))

lemmatized_frame = pd.DataFrame(list(zip(lemmatized_words)), columns =['Plt1Name'])
lemmatized_frame.Plt1Name.str.split(expand=True).stack().value_counts()




