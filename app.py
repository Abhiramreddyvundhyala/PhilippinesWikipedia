# 1. Page configuration MUST come first
import streamlit as st
st.set_page_config(
    page_title="Wikipedia Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Then import other libraries
import requests
from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# 3. Then add your custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stTextInput input {border-radius: 20px;}
    .stButton button {border-radius: 20px; background-color: #4CAF50; color: white;}
    .stAlert {border-radius: 10px;}
    .css-1aumxhk {background-color: #ffffff; border-radius: 10px; padding: 20px;}
    .positive {color: #28a745;}
    .negative {color: #dc3545;}
    .neutral {color: #6c757d;}
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🌏 Wikipedia Sentiment Analyzer")
st.markdown("Analyze sentiment of any Wikipedia article using machine learning and NLP techniques")

# Sidebar
with st.sidebar:
    st.header("Settings")
    default_topic = st.selectbox(
        "Default Topic",
        ["Philippines", "Artificial intelligence", "Climate change", "Basketball"],
        index=0
    )
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Enter a Wikipedia topic
    2. The app scrapes the article
    3. Analyzes sentiment using NLP
    4. Displays visualizations
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, NLTK, and Scikit-learn")

# Main content
tab1, tab2 = st.tabs(["Wikipedia Analysis", "Custom Text Analysis"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "🔍 Enter a Wikipedia topic to analyze:",
            value=default_topic,
            key="wiki_topic"
        )
    with col2:
        st.write("")
        analyze_btn = st.button("Analyze", key="analyze_btn")

    if analyze_btn:
        with st.spinner(f"Analyzing Wikipedia article about {topic}..."):
            try:
                # Web scraping
                url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                text = ' '.join([p.text for p in soup.find_all('p') if p.text.strip()])
                
                if not text:
                    st.warning("No content found for this topic. Please try another.")
                    st.stop()

                # Text cleaning
                text = re.sub(r'\[[0-9]*\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Tokenization
                sentences = [s for s in sent_tokenize(text) if len(s.split()) > 3]
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]

                # Sentiment analysis
                sentiments = []
                for sent in sentences:
                    blob = TextBlob(sent)
                    polarity = blob.sentiment.polarity
                    if polarity > 0.1:
                        sentiment = 'positive'
                    elif polarity < -0.1:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    sentiments.append(sentiment)
                
                sentiment_counts = Counter(sentiments)
                df = pd.DataFrame({'sentence': sentences, 'sentiment': sentiments})

                # Display results
                st.success(f"✅ Successfully analyzed: {topic}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution pie chart
                    st.subheader("Sentiment Distribution")
                    if sentiment_counts:
                        fig = px.pie(
                            values=list(sentiment_counts.values()),
                            names=list(sentiment_counts.keys()),
                            color=list(sentiment_counts.keys()),
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            hole=0.3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No sentiment data available")

                with col2:
                    # Sentiment bar chart
                    st.subheader("Sentiment Counts")
                    if sentiment_counts:
                        fig = px.bar(
                            x=list(sentiment_counts.keys()),
                            y=list(sentiment_counts.values()),
                            color=list(sentiment_counts.keys()),
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            labels={'x': 'Sentiment', 'y': 'Count'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No sentiment data available")

                # Word cloud
                st.subheader("Word Cloud")
                if filtered_words:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate(' '.join(filtered_words))
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("No words available for word cloud")

                # Top words
                st.subheader("Top 15 Frequent Words")
                if filtered_words:
                    word_freq = Counter(filtered_words).most_common(15)
                    words, counts = zip(*word_freq)
                    
                    fig = px.bar(
                        x=counts,
                        y=words,
                        orientation='h',
                        color=counts,
                        color_continuous_scale='Blues',
                        labels={'x': 'Frequency', 'y': 'Word'}
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No words available for frequency analysis")

                # Sample sentences
                st.subheader("Sample Sentences")
                sample_df = df.sample(min(5, len(df)))
                for _, row in sample_df.iterrows():
                    sentiment_class = row['sentiment']
                    st.markdown(
                        f"<div style='border-left: 4px solid; padding: 0.5em; margin: 0.5em 0; "
                        f"border-color: {'#28a745' if sentiment_class == 'positive' else '#dc3545' if sentiment_class == 'negative' else '#6c757d'};'>"
                        f"<span class='{sentiment_class}'><strong>{sentiment_class.upper()}</strong></span><br>"
                        f"{row['sentence'][:200]}{'...' if len(row['sentence']) > 200 else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error analyzing topic: {str(e)}")

with tab2:
    st.subheader("Analyze Custom Text")
    custom_text = st.text_area(
        "Enter your own text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    if st.button("Analyze Text", key="custom_analyze_btn"):
        if not custom_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    # Clean text
                    clean_text = re.sub(r'[^a-zA-Z. ]', ' ', custom_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
                    
                    if len(clean_text.split()) < 3:
                        st.warning("Please enter longer text (at least 3 words)")
                        st.stop()
                    
                    # Predict sentiment
                    features = vectorizer.transform([clean_text])
                    prediction = model.predict(features)[0]
                    proba = model.predict_proba(features)[0]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction Result")
                        if prediction == 'positive':
                            st.success("""
                            **Sentiment:** Positive 😊  
                            Confidence: {:.1f}%
                            """.format(proba[list(model.classes_).index('positive')] * 100))
                        else:
                            st.error("""
                            **Sentiment:** Negative 😞  
                            Confidence: {:.1f}%
                            """.format(proba[list(model.classes_).index('negative')] * 100))
                        
                        # TextBlob analysis for comparison
                        blob = TextBlob(clean_text)
                        st.markdown("""
                            **TextBlob Analysis:**  
                            - Polarity: {:.2f} (Range: -1 to 1)  
                            - Subjectivity: {:.2f} (Range: 0 to 1)
                            """.format(blob.sentiment.polarity, blob.sentiment.subjectivity))
                    
                    with col2:
                        st.subheader("Confidence Scores")
                        fig = go.Figure(go.Bar(
                            x=[f"{p*100:.1f}%" for p in proba],
                            y=model.classes_,
                            orientation='h',
                            marker_color=['#28a745', '#dc3545']
                        ))
                        fig.update_layout(
                            yaxis_title="Sentiment",
                            xaxis_title="Confidence",
                            height=200
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show processed text
                    with st.expander("Show processed text"):
                        st.write(clean_text)
                
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    Wikipedia Sentiment Analyzer • Data Science Project By Abhiram
    </div>
    """, unsafe_allow_html=True)
