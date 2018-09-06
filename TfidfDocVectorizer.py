from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import numpy as np
import nltk

class TfidfDocVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim

    def preprocess_doc(self, doc):
        stop_words = nltk.corpus.stopwords.words('english')
        words = [word.lower() for word in nltk.word_tokenize(doc) \
                 if word not in stop_words and word.isalpha()]
        return " ".join(words)
        
    def preprocess_docs(self, corpus):
        return [self.preprocess_doc(doc) for doc in corpus]
    
    def fit(self, corpus):
        corpus = self.preprocess_docs(corpus)
        tfidf = TfidfVectorizer(analyzer='word')
        tfidf.fit(corpus)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        
        return self

    def transform(self, corpus):
        corpus = self.preprocess_docs(corpus)
        doc_embeddings = []
        for words in corpus:
            word_embeddings = []
            for word in words.split():
                if word in self.word2vec:
                    word_embedding = self.word2vec[word]
                    word_embeddings.append(word_embedding)
                    
            doc_embedding = np.mean(np.array(word_embeddings), axis=0)
            doc_embeddings.append(doc_embedding)
        return np.array(doc_embeddings)
