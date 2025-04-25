import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from textblob import TextBlob

# Load your dataset
df = pd.read_csv(r"C:\Users\Win10\Desktop\muruga\spam.csv")
print("Sample Data:\n", df.head())

# Define manual stopwords
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
    'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
    'whom', 'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should',
    'now'
}

# Preprocess
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# df['clean_text'] = df['Review'].astype(str).apply(preprocess) default first coloumn name automatically takes
text_col = df.select_dtypes(include='object').columns[0]
df['clean_text'] = df[text_col].astype(str).apply(preprocess)

# WordCloud
all_words = " ".join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Cleaned Text")
plt.show()

# Most Common Words
all_tokens = " ".join(df['clean_text']).split()
word_freq = Counter(all_tokens)
print("\nMost Common Words:\n", word_freq.most_common(10))

word_freq_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Frequency', data=word_freq_df)
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45)
plt.show()

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # sklearn stopwords are built-in
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
tfidf_terms = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_dict = dict(zip(tfidf_terms, tfidf_scores))

# Show Top 10 TF-IDF Words
top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 TF-IDF terms:\n", top_tfidf)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Ranges from -1 (neg) to +1 (pos)

df['sentiment_score'] = df['clean_text'].apply(get_sentiment)

# Display sample
print("\nSentiment Scores Sample:\n", df[['Review', 'sentiment_score']].head())

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], kde=True, color='green')
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score (Range: -1 to 1)")
plt.ylabel("Frequency")
plt.show()

# Categorize sentiment based on score
def categorize_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(categorize_sentiment)

# Print sentiment label counts
print("\nSentiment Label Counts:\n", df['sentiment_label'].value_counts())

# Pie Chart
plt.figure(figsize=(6, 6))
df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgray'])
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()
