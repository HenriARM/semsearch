# Semantic search

![](./resources/SemanticSearch.png)

  \[
  \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
  \]

TODO: understnad baseline models and write description
TODO: add other metrics: ndcg, MRR
TODO: list models in eval and generate comparison table
TODO: finish 4th paragraph
TODO: kNN for fast vector search?
TODO: if you divide into passages it is one thing - you should do that for all models, otherwise each model will return closest text with different lengths

Methods:
TF-IDF
BM25
Word embeddings
Sentence embeddings (SBERT, OpenAI) - params: vector size, distance function, all different ways to split text
Siamese networks

3. Methods: Provide a short description of the methods that you use as well as a motivation for choosing these methods.

4. Analyses & Results: Describe how you apply the methods and present the results.

5. Conclusions: Briefly summarize your findings and the corresponding conclusions. Relate the outcomes to the research questions.

* For supervised problems, assess if your models have statistically significantly different performance.




## 1. Introduction
Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

The primary research question this study seeks to address is:
- How can semantic search models be effectively implemented to improve the relevance and accuracy of search results compared to traditional keyword-based search methods?

To delve deeper into the capabilities of semantic search, the following sub-questions will be explored:
- What are the key differences in performance between traditional search models (such as TF-IDF) and modern semantic models (like SBERT) in terms of precision, recall, and user satisfaction?
- How do various semantic search models manage the challenge of understanding complex queries in a domain-specific context?

This study is significant as it addresses the growing need for more sophisticated search mechanisms in various sectors, including academia, healthcare, legal, and customer service. By improving the efficiency and effectiveness of search technologies, organizations can enhance information accessibility and decision-making processes, ultimately leading to greater productivity and user satisfaction.

To answer these research questions, the study will:
- Review Existing Literature: Summarize current knowledge and theories related to semantic search.
Model Comparison: Implement and compare several models including a baseline model (TF-IDF) and advanced semantic models like SBERT.
- Dataset Selection and Preparation: Utilize datasets such as MS MARCO for testing and evaluating the models.
Performance Evaluation: Use metrics such as precision, recall, and Mean Reciprocal Rank (MRR) to assess the models.
- User Study: Conduct a user study to measure satisfaction and effectiveness from a human-centered perspective.

This report aims to provide a comprehensive analysis of semantic search technologies, focusing on the implementation and evaluation of advanced models, and offering insights into their practical implications and benefits.


## 2. Data

The dataset employed in this study is the MS MARCO (Microsoft Machine Reading Comprehension) dataset, specifically its v2.1 iteration. Developed by Microsoft, MS MARCO is designed to provide realistic information retrieval scenarios by presenting queries from real users and manually annotated passages from high-quality web pages. The dataset is widely utilized to train and evaluate machine learning models on their ability to comprehend and answer questions based on real-world text.

### Data Structure and Contents

MS MARCO consists of several key features:
- **Queries:** Real anonymized queries submitted by users to the Bing search engine.
- **Passages:** Collections of passages that are candidate answers to the queries. These passages are extracted from web pages and linked to their respective URLs.
- **Annotations:** Each passage is annotated with labels indicating whether it was selected as relevant to the corresponding query.
- **Query Types:** Categorized into informational, navigational, and transactional, providing insight into the user's intent.

### Relevance to the Problem

The relevance of MS MARCO to semantic search is significant:
- **Real-World Queries:** The dataset's use of actual search queries ensures that the models trained on it are better adapted to the nuances and variety of user intents in practical scenarios.
- **Diverse Content:** The variety in passage content and query types allows models to be tested across a broad spectrum of information needs and contexts, crucial for developing robust semantic search systems.

### Data Preparation

Before using the dataset for model training and evaluation, several preprocessing steps are necessary:
- **Tokenization:** Breaking down text into individual words or tokens.
- **Case Normalization:** Converting all text to lowercase to ensure uniformity in processing.
- **Punctuation Removal:** Stripping punctuation to reduce noise in the text data.
- **Stop Words Removal:** Eliminating common words that do not contribute significant information to the analysis.
- **Stemming or Lemmatization:** Reducing words to their base or root form to consolidate different forms of the same word.

These steps help in cleaning and standardizing the data, making it more amenable to processing by different semantic search models, whether they rely on classical text representations like TF-IDF or more advanced embeddings from neural networks.


## 3. Methods
In exploring the efficacy of semantic search, this study employs a range of methods from traditional information retrieval techniques to advanced machine learning models. The selected methods are chosen for their demonstrated ability in handling text data and improving search results through understanding contextual and semantic content.

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a statistical measure used to evaluate the importance of a word to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

- **Term Frequency (TF)** is calculated as:
  
  $
  \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
  $

- **Inverse Document Frequency (IDF)** is calculated as:

  $
  \text{IDF}(t, D) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents with term } t}\right)
  $

- **TF-IDF** is then computed by multiplying these two values:

  $
  \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
  $

TF-IDF is chosen for its simplicity and effectiveness in identifying relevant terms within a large dataset. It serves as a benchmark for comparing more complex algorithms like BM25.

### BM25 (Okapi BM25)

BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query, based on the query terms appearing in each document. It is an evolution of the TF-IDF model, incorporating probabilistic understanding of term occurrence, non-binary length normalization, and saturation.

- **BM25 Score** for a document $d$ given a query \( q \) is defined as:

  $
  \text{score}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
  $

  where $ f(t, d) $ is $ t $'s term frequency in the document \( d \), \( |d| \) is the length of the document, $ \text{avgdl} $ is the average document length in the text collection, \( k_1 \) and \( b \) are free parameters, usually chosen as \( k_1 = 2.0 \) and \( b = 0.75 \).

BM25 is selected for its robustness and has been shown to perform well across a wide range of text retrieval tasks. Its ability to handle various lengths of documents and the frequency of terms makes it a superior method for testing in complex semantic search scenarios.




Advanced Semantic Models

Word Embeddings (Word2Vec, GloVe)
Description: These models generate dense vector representations for words based on their contextual similarities. For semantic search, we compute the average of the word vectors in a query or document to create a single vector that represents the textual content.
Rationale: Word embeddings are chosen for their ability to capture deeper linguistic patterns and word associations, potentially improving the retrieval of semantically relevant documents.

BERT (Bidirectional Encoder Representations from Transformers)
Description: BERT processes words in relation to all other words in a sentence, unlike traditional models that read the text sequentially. This allows the model to interpret the full context of a word by looking at the words that come before and after it—ideal for understanding the intent behind search queries.
Rationale: BERT is integrated into the study for its state-of-the-art performance in a variety of NLP tasks, including its application in search scenarios where understanding the context and nuance of language is crucial.

## 4. Analyses & Result

Evaluation Metrics
Precision, Recall, and F1 Score: These metrics will assess the accuracy and relevancy of the search results provided by each model.
Mean Reciprocal Rank (MRR): This metric is used for evaluating the order in which the relevant documents are presented by the search algorithms.

TODO: write all formulas

## 5. Conclusions

## 6. References
https://microsoft.github.io/msmarco/
https://huggingface.co/datasets/microsoft/ms_marco


