
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load processed dataset
df = pd.read_csv("data/processed_resources.csv")

# Prepare searchable text
documents = df["searchable_text"].fillna("")

# Initialize TF-IDF
vectorizer = TfidfVectorizer()

# Convert text into numerical vectors
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Vectorization Completed!")
print("Number of documents:", tfidf_matrix.shape[0])
print("Number of features:", tfidf_matrix.shape[1])


# Search function
def search_resources(query):

    # Transform user query
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get top 5 results
    top_indices = similarity_scores.argsort()[::-1][:5]

    print("\nSearch Results:")

    for index in top_indices:

        if similarity_scores[index] > 0:

            print("\nTitle:", df.iloc[index]["title"])
            print("Topic:", df.iloc[index]["topic"])
            print("Similarity Score:",
                  round(similarity_scores[index], 4))


# User input
user_query = input("\nEnter your search query: ")

search_resources(user_query)
