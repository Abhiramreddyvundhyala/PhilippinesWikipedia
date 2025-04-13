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
This app analyzes sentiment based on a model trained on Wikipedia content for Argentina.
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
    
    # Prediction
    prediction = model.predict(user_vector)[0]
    proba = model.predict_proba(user_vector)[0]
    
    # Debug output
    st.write(f"Debug - Raw prediction value: {prediction}")
    st.write(f"Debug - Probabilities: {proba}")
    
    # FIXED: Correct sentiment mapping
    # Assuming class 0 is Negative and class 1 is Positive
    if prediction == 1:  # Positive
        sentiment_label = "Positive"
        sentiment_color = "green"
        positive_prob = proba[1]
        negative_prob = proba[0]
    else:  # Negative
        sentiment_label = "Negative"
        sentiment_color = "red"
        positive_prob = proba[1]
        negative_prob = proba[0]
    
    st.markdown(f"### 🎯 **Predicted Sentiment:** :{sentiment_color}[{sentiment_label}]")
    
    # Probability Bar Chart
    st.markdown("#### 📊 Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(["Negative", "Positive"], [negative_prob, positive_prob], color=['red', 'green'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    
    # TextBlob Analysis
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    st.markdown("#### 🧠 TextBlob Sentiment Analysis")
    st.write(f"- **Polarity:** `{polarity:.2f}`")
    st.write(f"- **Subjectivity:** `{subjectivity:.2f}`")

    # Seaborn Barplot for TextBlob
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=["Polarity", "Subjectivity"], y=[polarity, subjectivity], palette='coolwarm')
    ax2.set_ylim(-1, 1)
    ax2.set_title("TextBlob Sentiment Insights")
    st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("📘 Model trained on Philippines Wikipedia content using TextBlob + TF-IDF + SMOTE + Random Forest.")
st.markdown("👨‍💻 Created by *Abhiram Reddy*")
