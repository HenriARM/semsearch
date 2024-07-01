from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextSearchSBERT:
    def __init__(self, corpus):
        self.corpus = corpus
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a lightweight model for efficiency
        self.corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)

    def search(self, query):
        # Encode the query to get the query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities between the query embedding and the corpus embeddings
        cos_similarities = cosine_similarity([query_embedding], self.corpus_embeddings)[0]
        return cos_similarities


if __name__ == "__main__":
    # Example usage
    corpus = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "and runs away",
        "the quick brown fox jumps over the lazy dog",
        "BM25 is a ranking function"
    ]
    text_search = TextSearchSBERT(corpus)
    query = "quick fox"
    similarities = text_search.search(query)
    print(similarities)
