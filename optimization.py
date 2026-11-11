
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load processed dataset
df = pd.read_csv("data/processed_resources.csv")

documents = df["searchable_text"].fillna("")
resource_ids = df["id"].astype(str)

# 2. Ground truth: manually verified relevant resource IDs
ground_truth = {
    "Python programming": ["1", "3", "5", "7", "8"],
    "Data science": ["11", "13", "15", "17", "18"],
    "Machine learning": ["21", "23", "24", "25", "27"],
    "Deep learning": [
        "31", "32", "33", "34", "35",
        "36", "37", "38", "39", "40"
    ],
    "Natural Language Processing": [
        "41", "42", "43", "44", "45",
        "46", "47", "48", "49", "50"
    ]
}

# 3. Define configurations
configurations = {
    "Default TF-IDF": {},
    "Unigrams + Bigrams": {
        "ngram_range": (1, 2)
    },
    "Max DF = 0.8": {
        "max_df": 0.8
    },
    "Min DF = 3": {
        "min_df": 3
    }
}

# 4. Evaluate each configuration
k = 5

for config_name, parameters in configurations.items():

    print("\n" + "=" * 60)
    print("CONFIGURATION:", config_name)
    print("=" * 60)

    vectorizer = TfidfVectorizer(**parameters)
    tfidf_matrix = vectorizer.fit_transform(documents)

    print("Vocabulary size:", len(vectorizer.vocabulary_))

    precisions = []
    recalls = []
    f1_scores = []

    for query, relevant_ids in ground_truth.items():

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            tfidf_matrix
        ).flatten()

        ranked_indices = scores.argsort()[::-1]

        retrieved_indices = [
            i for i in ranked_indices if scores[i] > 0
        ][:k]

        retrieved_ids = [
            resource_ids.iloc[i] for i in retrieved_indices
        ]

        relevant_retrieved = len(
            set(retrieved_ids) & set(relevant_ids)
        )

        precision = relevant_retrieved / k
        recall = relevant_retrieved / len(relevant_ids)

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print("\nQuery:", query)
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1-score:", round(f1, 3))

    print("\n--- MACRO AVERAGES ---")
    print("Average Precision:", round(sum(precisions) / len(precisions), 3))
    print("Average Recall:", round(sum(recalls) / len(recalls), 3))
    print("Average F1-score:", round(sum(f1_scores) / len(f1_scores), 3))
