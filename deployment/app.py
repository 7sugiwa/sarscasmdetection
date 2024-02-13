import streamlit as st
from prediction import predict_sarcasm
import pandas as pd
from eda import generate_wordcloud, plot_top_ngrams, calculate_sentiment, plot_top_tfidf_terms_modified, plot_flesch_reading_ease

# Load the dataset
df = pd.read_csv('a.csv')

st.title('Sarcasm Detection')

# Model inference section
st.write('Enter a headline to check if it is sarcastic or not:')
user_input = st.text_input("Headline")
if user_input:
    prediction = predict_sarcasm(user_input)
    st.write('The headline is sarcastic.' if prediction == 1 else 'The headline is not sarcastic.')

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

# Word Cloud Analysis
st.subheader("Word Cloud Analysis")
fig_wc = generate_wordcloud(df)
st.pyplot(fig_wc)
st.write("""
**Sarcastic Headlines**: Frequent words like "new," "man," "nation" suggest a focus on current events with a satirical twist.
         
**Non-Sarcastic Headlines**: Words such as "Trump" and "new" indicate a focus on current affairs, approached more straightforwardly.
""")

# N-Gram Analysis
st.subheader("N-Gram Analysis")
headlines_list = df['headline'].tolist()
fig_ngram = plot_top_ngrams(headlines_list, n=2, top_k=10, title="Top Bi-grams in Headlines")
st.pyplot(fig_ngram)
st.write("""
**Bigrams**: Phrases like "Donald Trump," "Hillary Clinton," and "White House" are prevalent in both, but likely used differently in sarcastic contexts.
""")

# Sentiment Analysis
st.subheader("Sentiment Analysis")
fig_sentiment = calculate_sentiment(df['headline'])  # Pass the list of headlines
st.pyplot(fig_sentiment)
st.write("""
         **Overall Similar Distribution**: Similar sentiment distributions in both categories, with a subtle use of positive language for negative or ironic sentiments in sarcastic headlines.
""")

# TF-IDF Analysis
st.subheader("TF-IDF Analysis")
fig_tfidf = plot_top_tfidf_terms_modified(df['headline'])  # Ensure this function is defined to handle the data
st.pyplot(fig_tfidf)
st.write("""
Similar results to the N-Gram Analysis, indicating top topics.
""")

# Flesch Reading Ease Score Analysis
st.subheader("Flesch Reading Ease Score Analysis")
fig_flesch = plot_flesch_reading_ease(df) 
st.pyplot(fig_flesch)
st.write("""
**Similar Scores with Slight Variance**: Comparable complexity in language use, with non-sarcastic headlines being slightly more straightforward.
""")
