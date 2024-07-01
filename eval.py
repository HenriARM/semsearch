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


def evaluate(dataset, model, num_samples=100):
    results = []
    sum_rr = 0  # Sum of Reciprocal Ranks for MRR calculation

    for i in range(num_samples):
        query = dataset["query"][i]
        # TODO
        # passages = [p['passage_text'] for p in dataset["passages"][i]]
        # selected = [p['is_selected'] for p in dataset["passages"][i]]
        passages = dataset["passages"][i]["passage_text"]
        selected = dataset["passages"][i]["is_selected"]
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
    mrr = sum_rr / num_samples
    return avg_precision, avg_recall, avg_f1, mrr



corpus = ds["train"][:1000]["passages"][0]["passage_text"]
# corpus = [p for entry in ds["train"][:100]["passages"] for p in entry["passage_text"]]
# model = TextSearchSBERT(corpus)
# model = BM25(corpus)
model = TFIDF(corpus)
avg_precision, avg_recall, avg_f1, mrr = evaluate(ds["train"][:100], model)
print(
    f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1 Score: {avg_f1}, MRR: {mrr}"
)
