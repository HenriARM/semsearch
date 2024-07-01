from datasets import load_dataset
from bm25 import BM25
from tfidf import TFIDF
from sbert import TextSearchSBERT

ds = load_dataset("microsoft/ms_marco", "v2.1")


def compute_metrics(pred, true):
    tp = len(set(pred) & set(true))
    precision = tp / len(pred) if pred else 0
    recall = tp / len(true) if true else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Calculate Reciprocal Rank
    rr = 0.0
    for rank, index in enumerate(pred, start=1):
        if index in true:
            rr = 1.0 / rank
            break

    return precision, recall, f1, rr


def evaluate(dataset):
    results = []
    sum_rr = 0

    for i in range(len(dataset)):
        query = dataset["query"][i]
        passage = dataset["passages"][i]["passage_text"]
        selected = dataset["passages"][i]["is_selected"]
        model = TFIDF(passage)
        scores = model.search(query)
        predicted_indices = sorted(
            range(len(scores)), key=lambda k: scores[k], reverse=True
        )
        ground_truth_indices = [
            idx for idx, is_sel in enumerate(selected) if is_sel == 1
        ]
        precision, recall, f1, rr = compute_metrics(
            predicted_indices[:5], ground_truth_indices
        )
        results.append((precision, recall, f1))
        sum_rr += rr

    avg_precision = sum(r[0] for r in results) / len(results)
    avg_recall = sum(r[1] for r in results) / len(results)
    avg_f1 = sum(r[2] for r in results) / len(results)
    mrr = sum_rr / len(dataset)
    return avg_precision, avg_recall, avg_f1, mrr


data_len = 100
# model = TextSearchSBERT(corpus)
# model = BM25(corpus)
avg_precision, avg_recall, avg_f1, mrr = evaluate(ds["train"][:data_len])
print(
    f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}, MRR: {mrr}"
)
