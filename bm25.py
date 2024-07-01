import math
from collections import Counter

class BM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.document_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.document_lengths) / len(self.corpus)
        self.doc_freqs = []
        self.idf = {}
        self.initialize()

    def initialize(self):
        """
        Initialize document frequencies and inverse document frequencies.
        """
        df = {}
        for document in self.corpus:
            # Tokenize document, here split by whitespace
            tokens = document.split()
            # Count frequencies within this document
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            # Calculate document frequency for IDF calculation
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in df:
                    df[token] += 1
                else:
                    df[token] = 1
        
        total_documents = len(self.corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((total_documents - freq + 0.5) / (freq + 0.5) + 1)
    
    def doc_score(self, document, query):
        """
        Computes the BM25 score of a single document for a given query.
        """
        score = 0.0
        doc_length = len(document.split())
        frequencies = Counter(document.split())
        for word in query.split():
            if word in frequencies:
                frequency = frequencies[word]
                idf = self.idf.get(word, 0)
                score += (idf * frequency * (1.5 + 1)) / (frequency + 1.5 * (1 - 0.75 + 0.75 * (doc_length / self.avg_doc_length)))
        return score

    def search(self, query):
        """
        Return a list of scores for all documents in the corpus.
        """
        scores = []
        for document in self.corpus:
            scores.append(self.doc_score(document, query))
        return scores

# Example usage
corpus = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "and runs away",
    "the quick brown fox jumps over the lazy dog",
    "BM25 is a ranking function"
]
bm25 = BM25(corpus)
query = "quick fox"
scores = bm25.search(query)
print(scores)
