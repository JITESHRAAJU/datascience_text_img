import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from textblob import TextBlob

# Load dataset
df = pd.read_csv("/mnt/data/c202d947-4cef-47b8-b1c4-678a6eb3d3d0.csv", encoding='ISO-8859-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']


# Stopwords
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
} # keep your existing stop_words set

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].astype(str).apply(preprocess)

# WordCloud
all_words = " ".join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Cleaned Text")
plt.show()

# Common Words
all_tokens = all_words.split()
word_freq = Counter(all_tokens)
print("\nMost Common Words:\n", word_freq.most_common(10))

word_freq_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Frequency', data=word_freq_df)
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45)
plt.show()

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
tfidf_terms = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_dict = dict(zip(tfidf_terms, tfidf_scores))

top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 TF-IDF terms:\n", top_tfidf)

# Sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['clean_text'].apply(get_sentiment)

# Sample output
print("\nSentiment Scores Sample:\n", df[['text', 'sentiment_score']].head())

# Sentiment distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], kde=True, color='green')
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()

# Labeling
def categorize_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(categorize_sentiment)
print("\nSentiment Label Counts:\n", df['sentiment_label'].value_counts())

# Pie Chart
plt.figure(figsize=(6, 6))
df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgray'])
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()
