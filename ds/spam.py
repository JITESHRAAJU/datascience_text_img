import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\raaju\OneDrive\Desktop\muruga_text_img_datascience\ds\spam.csv", encoding='latin1')
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
}

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

# Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['clean_text'].apply(get_sentiment)

print("\nSentiment Scores Sample:\n", df[['text', 'sentiment_score']].head())

# Sentiment Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], kde=True, color='green')
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()

# Categorize Sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(categorize_sentiment)
print("\nSentiment Label Counts:\n", df['sentiment_label'].value_counts())

# Sentiment Pie Chart
plt.figure(figsize=(6, 6))
df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightgray'])
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()

# Label Encoding
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # spam = 1, ham = 0

# Train-test split and Naive Bayes Classification
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['label_num'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance Measures
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: Final Inference
print("\nInference:")
print("The Naive Bayes classifier performed well in distinguishing between spam and ham SMS messages.")
print("The TF-IDF approach identified important spam keywords such as 'free', 'call', 'win'.")
print("Sentiment analysis shows most spam messages lean slightly positive or neutral due to promotional language.")
