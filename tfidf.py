import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.corpus = corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()

    def doc_score(self, document_index, query):
        """
        Computes the cosine similarity of a given document with the query.
        """
        # Transform the query to the same vector space as the corpus
        query_vec = self.vectorizer.transform([query]).toarray()

        # Compute the cosine similarity between the document vector and query vector
        document_vec = self.tfidf_matrix[document_index]
        dot_product = np.dot(query_vec, document_vec)
        norm_product = np.linalg.norm(query_vec) * np.linalg.norm(document_vec)
        cosine_similarity = dot_product / norm_product if norm_product != 0 else 0

        if cosine_similarity == 0:
            return 0
        else:
            return float(cosine_similarity.flatten()[0])

    def search(self, query):
        """
        Return the cosine similarity scores for all documents in the corpus relative to a given query.
        """
        scores = [self.doc_score(i, query) for i in range(len(self.corpus))]
        # scores = [float(score) for score in scores]
        return scores


if __name__ == "__main__":
    # Example usage
    corpus = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "and runs away",
        "the quick brown fox jumps over the lazy dog",
        "TF-IDF is a ranking function",
    ]
    tfidf = TFIDF(corpus)
    query = "quick fox"
    scores = tfidf.search(query)
    print(scores)
