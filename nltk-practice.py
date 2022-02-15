import nltk
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = set(stopwords.words("english"))

example_string = """
    Muad'Dib learned rapidly because his first training was in how to learn.
    And the first lesson of all was the basic trust that he could learn.
    It's shocking to find how many people do not believe they can learn,
    and how many more believe learning to be difficult."""

string_for_stemming = """
... The crew of the USS Discovery discovered many discoveries.
... Discovering is what explorers do."""

POS_string = """
... If you wish to make an apple pie from scratch,
... you must first invent the universe."""

lemmatizing_string = "The friends of DeSoto love scarves."

def main():
    sentence_tok = sent_tokenize(example_string)
    # print_array(sentence_tok)

    word_tok = word_tokenize(example_string)
    # print_array(word_tok)
    filtered = filter_stop_words(word_tok)
    # print_array(filtered)


    # STEMMING ==================================================
    stemmer = PorterStemmer()
    stem_words = word_tokenize(string_for_stemming)
    # print_array(stem_words)
    # print()
    stemmed_words = [stemmer.stem(word) for word in stem_words]
    # print_array(stemmed_words)

    # PART OF SENTENCE ==========================================
    pos_words = word_tokenize(POS_string)
    # print_array(pos_words)
    # print()
    pos_tagged_words = nltk.pos_tag(pos_words)
    # print_array(pos_tagged_words)

    # LEMMATIZING ===============================================
    lemmatizer = WordNetLemmatizer()
    lemma_words = word_tokenize(lemmatizing_string)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in lemma_words]
    # print_array(lemmatized_words)
    print(lemmatizer.lemmatize("worst", pos="a"))


def print_array(array):
    i = 1
    for el in array:
        print("%d: %s" % (i, el))
        i += 1

def filter_stop_words(words):
    filtered_list = []
    for word in words:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    return filtered_list

def sexy_filter_stop_words(words):
    return [ word for word in words if word.casefold() not in stop_words ]

if __name__ == "__main__":
    main()
