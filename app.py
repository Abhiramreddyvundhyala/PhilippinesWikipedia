import streamlit as st
import joblib
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Debug: Check model classes and features
try:
    st.write("### Model Debug Info")
    st.write(f"Model classes: {model.classes_}")
    st.write(f"Number of features: {len(vectorizer.get_feature_names_out())}")
except Exception as e:
    st.warning(f"Couldn't get model details: {e}")

# App Interface
st.title("🌎 Philippines Wikipedia Sentiment Analysis")
st.subheader("Powered by Random Forest Classifier")

# Default examples
positive_example = "Philippines has beautiful beaches and friendly people"
negative_example = "The worst experience ever, terrible service"

# Input Text
user_input = st.text_area("✍️ Enter your sentence here:", positive_example)

# Prediction Function
def analyze_sentiment(text):
    try:
        # Vectorize input
        text_vector = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # Get class order (handle different sklearn versions)
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if 1 in classes and 0 in classes:
                positive_idx = np.where(classes == 1)[0][0]
                negative_idx = np.where(classes == 0)[0][0]
            else:  # Handle [-1, 1] systems
                positive_idx = np.where(classes == 1)[0][0] if 1 in classes else 1
                negative_idx = np.where(classes == -1)[0][0] if -1 in classes else 0
        else:
            positive_idx = 1
            negative_idx = 0
        
        return {
            'prediction': prediction,
            'positive_prob': probabilities[positive_idx],
            'negative_prob': probabilities[negative_idx],
            'is_positive': prediction == classes[positive_idx] if hasattr(model, 'classes_') else prediction == 1
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Predict Button
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze")
    else:
        # Show Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_input)
        st.image(wordcloud.to_array(), caption="Word Cloud", use_container_width=True)
        
        # Analyze sentiment
        result = analyze_sentiment(user_input)
        if result:
            # Display results
            st.write("### Prediction Details")
            st.write(f"Raw prediction value: {result['prediction']}")
            st.write(f"Positive probability: {result['positive_prob']:.4f}")
            st.write(f"Negative probability: {result['negative_prob']:.4f}")
            
            # Determine sentiment
            sentiment_label = "Positive" if result['is_positive'] else "Negative"
            sentiment_color = "green" if result['is_positive'] else "red"
            
            st.markdown(f"## 🎯 **Predicted Sentiment:** :{sentiment_color}[{sentiment_label}]")
            
            # Probability chart
            fig, ax = plt.subplots()
            ax.bar(["Negative", "Positive"], 
                  [result['negative_prob'], result['positive_prob']], 
                  color=['red', 'green'])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            st.pyplot(fig)
            
            # TextBlob analysis
            blob = TextBlob(user_input)
            st.write("### TextBlob Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Polarity", f"{blob.sentiment.polarity:.2f}", 
                         help="-1 (negative) to 1 (positive)")
            with col2:
                st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}",
                         help="0 (objective) to 1 (subjective)")

# Test Cases Section
with st.expander("🧪 Test Cases"):
    st.write("Try these test sentences:")
    if st.button("Positive Example"):
        st.session_state.user_input = positive_example
    if st.button("Negative Example"):
        st.session_state.user_input = negative_example

# Footer
st.markdown("---")
st.markdown("📘 Model trained on Wikipedia content using TF-IDF and Random Forest")
