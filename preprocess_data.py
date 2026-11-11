
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

# Download English stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("data/learning_resources.csv")

# Get English stopwords
stop_words = set(stopwords.words("english"))


# Text cleaning function
def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back together
    return " ".join(words)


# Apply preprocessing to selected columns
columns_to_clean = ["title", "topic", "resource_type", "summary"]

for column in columns_to_clean:
    df[column + "_clean"] = df[column].apply(clean_text)


# Combine cleaned text for future searching
df["searchable_text"] = (
    df["title_clean"] + " " +
    df["topic_clean"] + " " +
    df["resource_type_clean"] + " " +
    df["summary_clean"]
)

# Display results
print(df[["title", "title_clean", "searchable_text"]].head())

# Save processed dataset
df.to_csv("data/processed_resources.csv", index=False)

print("\nPreprocessing completed!")
print("Total records:", len(df))
print("Processed dataset saved!")
