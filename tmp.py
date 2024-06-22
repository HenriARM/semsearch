import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK resources downloaded
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["data"]


def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [
        word for word in words if word not in stop_words and word.isalnum()
    ]
    return filtered_words


def find_answer(question, context):
    question_words = set(preprocess_text(question))
    context_words = preprocess_text(context)
    answer = []
    for word in context_words:
        if word in question_words:
            answer.append(word)
    return " ".join(answer)


# Example usage
data = load_data("train-v1.1.json")  # Adjust the filename/path as needed

# Test on a small portion of the dataset
sample = data[0]["paragraphs"][0]
context = sample["context"]
questions = sample["qas"]

for qa in questions[:5]:  # Let's test on the first 5 questions
    question = qa["question"]
    answer = qa["answers"][0]["text"]
    predicted_answer = find_answer(question, context)
    print(f"Question: {question}")
    print(f"Answer {answer}")
    print(f"Predicted Answer: {predicted_answer}\n")
