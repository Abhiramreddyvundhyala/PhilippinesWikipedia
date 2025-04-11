import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Page Setup
st.set_page_config(page_title="Philippines Wiki Sentiment Analyzer", page_icon="🇵🇭", layout="wide")

# Load Model and Vectorizer
model = joblib.load("random_forest_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI Header
st.title("Wikipedia Sentiment Analyzer for Philippines Topics")
st.markdown("Analyze the **sentiment of any Wikipedia topic** – especially those related to the **Philippines** – using Machine Learning, NLP, and Visualizations.")

# User Input
topic = st.text_input("🔎 Enter a Wikipedia topic (e.g., Philippines, Manny Pacquiao, Cebu):", value="Philippines")

if st.button("Fetch & Analyze"):
    try:
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join([p.text for p in soup.find_all("p")])

        # Clean text
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"[^a-zA-Z. ]", "", text)
        text = text.lower()

        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

        # Sentiment via TextBlob
        def get_sentiment(sent):
            polarity = TextBlob(sent).sentiment.polarity
            if polarity > 0:
                return "positive"
            elif polarity < 0:
                return "negative"
            else:
                return "neutral"

        sentiments = [get_sentiment(s) for s in sentences]
        sentiment_counts = Counter(sentiments)

        # 📊 Sentiment Pie Chart
        st.subheader("📊 Sentiment Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90,
                colors=['lightgreen', 'lightcoral', 'lightgray'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # ☁️ WordCloud
        st.subheader("☁️ WordCloud of Cleaned Wikipedia Text")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

        # 📶 Top Frequent Words
        st.subheader("📶 Top 15 Frequent Words")
        freq = Counter(filtered_words).most_common(15)
        words_, counts_ = zip(*freq)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=list(words_), y=list(counts_), ax=ax3, palette="Blues_r")
        ax3.set_ylabel("Frequency")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        st.pyplot(fig3)

        # 📋 Sample Sentences
        st.subheader("📋 Sample Sentences & Sentiments")
        for s, sent in zip(sentences[:5], sentiments[:5]):
            st.markdown(f"- **{sent.upper()}**: {s}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# 💬 Custom Input
st.markdown("---")
st.subheader("💬 Predict Sentiment of Your Own Sentence")

user_input = st.text_area("Enter your sentence below:")

if st.button("Predict My Sentiment"):
    if user_input.strip() == "":
        st.warning("Please type a sentence to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment_label}**")
