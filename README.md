# Semantic search

TODO: kNN for fast vector search?
TODO: if you divide into passages it is one thing - you should do that for all models, otherwise each model will return closest text with different lengths

Methods:
TF-IDF
BM25
Word embeddings
Sentence embeddings (SBERT, OpenAI) - params: vector size, distance function, all different ways to split text


1. Introduction: Introduce the research questions, why it is interesting and how you are going to answer the research questions.

2. Data: Briefly explain the data that you have, where they are from and why they are relevant for your problem. If appropriate, describe relevant data properties and general data preparation steps.

3. Methods: Provide a short description of the methods that you use as well as a motivation for choosing these methods.

4. Analyses & Results: Describe how you apply the methods and present the results.

5. Conclusions: Briefly summarize your findings and the corresponding conclusions. Relate the outcomes to the research questions.

6. References
* For supervised problems, assess if your models have statistically significantly different performance.




## 1. Introduction
Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

Research Questions
The primary research question this study seeks to address is:

How can semantic search models be effectively implemented to improve the relevance and accuracy of search results compared to traditional keyword-based search methods?
To delve deeper into the capabilities of semantic search, the following sub-questions will be explored:

What are the key differences in performance between traditional search models (such as TF-IDF) and modern semantic models (like SBERT) in terms of precision, recall, and user satisfaction?
How do various semantic search models manage the challenge of understanding complex queries in a domain-specific context?
Significance of the Study
This study is significant as it addresses the growing need for more sophisticated search mechanisms in various sectors, including academia, healthcare, legal, and customer service. By improving the efficiency and effectiveness of search technologies, organizations can enhance information accessibility and decision-making processes, ultimately leading to greater productivity and user satisfaction.

Methodological Approach
To answer these research questions, the study will:

Review Existing Literature: Summarize current knowledge and theories related to semantic search.
Model Comparison: Implement and compare several models including a baseline model (TF-IDF) and advanced semantic models like SBERT.
Dataset Selection and Preparation: Utilize datasets such as MS MARCO for testing and evaluating the models.
Performance Evaluation: Use metrics such as precision, recall, and Mean Reciprocal Rank (MRR) to assess the models.
User Study: Conduct a user study to measure satisfaction and effectiveness from a human-centered perspective.
This report aims to provide a comprehensive analysis of semantic search technologies, focusing on the implementation and evaluation of advanced models, and offering insights into their practical implications and benefits.


## 2.



## 3. Methods
In exploring the efficacy of semantic search, this study employs a range of methods from traditional information retrieval techniques to advanced machine learning models. The selected methods are chosen for their demonstrated ability in handling text data and improving search results through understanding contextual and semantic content.

Traditional Search Model: TF-IDF + Cosine Similarity
Description: Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus. It is often used in conjunction with cosine similarity, which measures the cosine of the angle between two vectors. This approach serves as our baseline.
Rationale: TF-IDF is selected as the baseline for its widespread use and effectiveness in traditional search applications, providing a benchmark for evaluating more advanced semantic models.

Advanced Semantic Models

Word Embeddings (Word2Vec, GloVe)
Description: These models generate dense vector representations for words based on their contextual similarities. For semantic search, we compute the average of the word vectors in a query or document to create a single vector that represents the textual content.
Rationale: Word embeddings are chosen for their ability to capture deeper linguistic patterns and word associations, potentially improving the retrieval of semantically relevant documents.

BERT (Bidirectional Encoder Representations from Transformers)
Description: BERT processes words in relation to all other words in a sentence, unlike traditional models that read the text sequentially. This allows the model to interpret the full context of a word by looking at the words that come before and after it—ideal for understanding the intent behind search queries.
Rationale: BERT is integrated into the study for its state-of-the-art performance in a variety of NLP tasks, including its application in search scenarios where understanding the context and nuance of language is crucial.

Evaluation Metrics
Precision, Recall, and F1 Score: These metrics will assess the accuracy and relevancy of the search results provided by each model.
Mean Reciprocal Rank (MRR): This metric is used for evaluating the order in which the relevant documents are presented by the search algorithms.

TODO: write all formulas

## 6. References
https://microsoft.github.io/msmarco/
https://huggingface.co/datasets/microsoft/ms_marco


