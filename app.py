import streamlit as st
import joblib
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# App Title
st.title("🌎 Philippines Wikipedia Sentiment Analysis")
st.subheader("Powered by Random Forest Classifier")

st.markdown("""
This app analyzes sentiment based on a model trained on Wikipedia content.
""")

# Default sentence
default_sentence = "Philippines is a stunning country with passionate people and rich culture."

# Input Text
user_input = st.text_area("✍️ Enter your sentence here:", default_sentence)

# Show Word Cloud
if user_input:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
    st.image(wordcloud.to_array(), caption="Word Cloud of Your Input", use_container_width=True)

# Predict Button
if st.button("🔍 Analyze Sentiment"):
    # TF-IDF Transformation
    user_vector = vectorizer.transform([user_input])
    
    # Get prediction and probabilities
    prediction = model.predict(user_vector)[0]
    proba = model.predict_proba(user_vector)[0]
    
    # FIXED: Ensure consistent interpretation
    # We'll use the higher probability to determine sentiment
    if proba[1] > proba[0]:  # Positive has higher probability
        final_sentiment = "Positive"
        sentiment_color = "green"
    else:
        final_sentiment = "Negative"
        sentiment_color = "red"
    
    # Display results
    st.markdown(f"### 🎯 **Predicted Sentiment:** :{sentiment_color}[{final_sentiment}]")
    
    # Probability Bar Chart (now always matches the label)
    st.markdown("#### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    bars = ax.bar(["Negative", "Positive"], proba, color=['red', 'green'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # TextBlob Analysis
    blob = TextBlob(user_input)
    st.markdown("#### 🧠 TextBlob Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Polarity", f"{blob.sentiment.polarity:.2f}", 
                 help="-1 (negative) to 1 (positive)")
    with col2:
        st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}",
                 help="0 (objective) to 1 (subjective)")

# Footer
st.markdown("---")
st.markdown("📘 Model trained on Wikipedia content using TextBlob + TF-IDF + Random Forest")
st.markdown("👨‍💻 Created by *Abhiram Reddy*")
