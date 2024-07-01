# Semantic search

![](./resources/SemanticSearch.png)

TODO: list models in eval and generate comparison table

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

### SBERT (Sentence-BERT)

Sentence-BERT (SBERT) is a modification of the pre-trained BERT network that uses siamese and triplet network structures to produce embeddings that are specifically tuned for semantic similarity comparison. This model is highly effective for semantic search applications because it generates sentence embeddings that can be directly compared using cosine similarity, making it significantly faster for semantic comparisons than typical BERT models.

The SBERT model utilized in this study is instantiated with the `all-MiniLM-L6-v2` model, a lightweight version of SBERT optimized for greater speed and lower resource consumption while maintaining strong performance.

Unlike TF-IDF and BM25, which rely on term frequency metrics and ignore word order and semantics, SBERT understands the context and meaning behind sentences. This leads to significantly improved performance in matching queries with relevant texts based on semantic content rather than mere keyword overlap.


## 4. Analyses & Result

To comprehensively assess the performance of the semantic search models, several metrics are used: Precision, Recall, F1 Score, and Mean Reciprocal Rank (MRR). Each of these metrics helps to provide insights into different aspects of model effectiveness.

### Precision

Precision measures the accuracy of the returned results by calculating the proportion of relevant documents retrieved over the total number of documents retrieved.

- **Formula**:
  
  \[
  \text{Precision} = \frac{\text{Number of Relevant Documents Retrieved}}{\text{Total Number of Documents Retrieved}}
  \]

### Recall

Recall measures the ability of the model to retrieve all relevant documents by calculating the proportion of relevant documents retrieved over the total number of relevant documents available.

- **Formula**:
  
  \[
  \text{Recall} = \frac{\text{Number of Relevant Documents Retrieved}}{\text{Total Number of Relevant Documents}}
  \]

### F1 Score

The F1 Score is the harmonic mean of Precision and Recall, providing a balance between them. It is particularly useful when the contribution of both precision and recall is equally important.

- **Formula**:
  
  \[
  \text{F1 Score} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

### Mean Reciprocal Rank (MRR)

Mean Reciprocal Rank is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness. The MRR provides a measure of the effectiveness of a semantic search algorithm, specifically focusing on the rank of the first correct answer found.

- **Formula**:
  
  \[
  \text{MRR} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\text{rank}_i}
  \]

where \(Q\) is the number of queries, and \(\text{rank}_i\) is the position of the first relevant document in the list of returned documents for the \(i\)-th query.

These metrics are crucial for understanding both the effectiveness and efficiency of the semantic search models in retrieving relevant information based on user queries.


## 5. Conclusions

## 6. References
https://microsoft.github.io/msmarco/
https://huggingface.co/datasets/microsoft/ms_marco


