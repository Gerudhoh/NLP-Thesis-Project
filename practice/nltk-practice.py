import nltk
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("book")
from nltk.book import *
from nltk import FreqDist
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

lotr_quote = "It's a dangerous business, Frodo, going out your door."

quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

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
    # print(lemmatizer.lemmatize("worst", pos="a"))

    # CHUNKING =================================================
    lotr_words = word_tokenize(lotr_quote)
    lotr_pos_tags = nltk.pos_tag(lotr_words)
    # print_array(lotr_pos_tags)
    chu_grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = nltk.RegexpParser(chu_grammar)
    tree = chunk_parser.parse(lotr_pos_tags)
    # tree.draw()

    # CHINKING =================================================
    chi_grammar = """
    Chunk: {<.*>+}
       }<JJ>{"""
    chink_parser = nltk.RegexpParser(chi_grammar)
    tree = chink_parser.parse(lotr_pos_tags)
    # tree.draw()

    # NAMED ENTITY RECOGNITION ================================
    tree = nltk.ne_chunk(lotr_pos_tags)
    # tree.draw()
    nes = extract_ne(quote)
    # print_array(nes)

    ## Concordance =============================================
    # print("Men")
    # text8.concordance("man")
    # print("Women")
    # text8.concordance("woman")

    # Frequency Distro =========================================
    frequency_distribution = FreqDist(text8)
    # print(frequency_distribution)
    # print_array(frequency_distribution.most_common(20))
    # Get rid of stop words:
    meaningful_words = [ word for word in text8 if word.casefold() not in stop_words ]
    frequency_distribution = FreqDist(meaningful_words)
    # print(frequency_distribution)
    # print_array(frequency_distribution.most_common(20))

    # Collocations ============================================
    text8.collocations()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
    # But to do collocations and such, you need to convert the data type
    # From an array into a nltk text object
    lemmatized_text = nltk.Text(lemmatized_words)
    lemmatized_text.collocations()

def extract_ne(quote):
    words = word_tokenize(quote)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

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
