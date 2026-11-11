
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load processed dataset
df = pd.read_csv("data/processed_resources.csv")

id_column = "id"

# 2. Prepare documents
documents = df["searchable_text"].fillna("")

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


# 4. Ground truth: manually verified relevant IDs
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



print("\nDEBUG: Ground Truth")
print(ground_truth["Python programming"])
print(ground_truth["Data science"])


# 5. Evaluation function
def evaluate_search(query, relevant_ids, k=5):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Rank results from highest to lowest
    ranked_indices = scores.argsort()[::-1]

    print("DEBUG: k =", k)

    # Retrieve top-k results with positive similarity
    retrieved_indices = [
        i for i in ranked_indices if scores[i] > 0
    ][:k]

    retrieved_ids = [
        str(df.iloc[i][id_column])
        for i in retrieved_indices
    ]

    relevant_ids = set(str(x) for x in relevant_ids)

    retrieved_id_set = set(retrieved_ids)

    # Calculate relevant retrieved documents
    relevant_retrieved = len(
        retrieved_id_set.intersection(relevant_ids)
    )

    # Precision
    precision = (
        relevant_retrieved / len(retrieved_ids)
        if retrieved_ids else 0
    )

    # Recall
    recall = (
        relevant_retrieved / len(relevant_ids)
        if relevant_ids else 0
    )

    # F1-score
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )


    
    # Display evaluation results
    print("\n" + "=" * 60)
    print("Query:", query)
    print("Retrieved IDs:", retrieved_ids)
    print("Relevant retrieved:", relevant_retrieved)

    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1-score:", round(f1, 3))

    print("\nRetrieved Resources:")

    for index in retrieved_indices:
        print(
            "ID:", df.iloc[index][id_column],
            "| Title:", df.iloc[index]["title"],
            "| Score:", round(scores[index], 4)
        )
    


# 6. Run evaluation for all five queries
for query, relevant_ids in ground_truth.items():
    evaluate_search(query, relevant_ids, k=10)


# 7. Dataset diagnostics
print("\n" + "=" * 60)
print("DATASET DIAGNOSTICS")

print(
    "Duplicate searchable text:",
    df["searchable_text"].duplicated().sum()
)

print("\nRepeated titles:")
print(df["title"].value_counts().head(10))

