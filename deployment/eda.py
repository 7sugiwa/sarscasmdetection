import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
import textstat

def generate_wordcloud(df):
    # Wordcloud for Sarcastic Headlines
    sarcastic_headlines = ' '.join(df[df['is_sarcastic'] == 1]['headline'])
    wordcloud_sarcastic = WordCloud(width=800, height=400, background_color='white').generate(sarcastic_headlines)

    # Wordcloud for Non-Sarcastic Headlines
    non_sarcastic_headlines = ' '.join(df[df['is_sarcastic'] == 0]['headline'])
    wordcloud_non_sarcastic = WordCloud(width=800, height=400, background_color='white').generate(non_sarcastic_headlines)

    # Create a figure to display the word clouds
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Display word cloud for sarcastic headlines
    axs[0].imshow(wordcloud_sarcastic, interpolation='bilinear')
    axs[0].set_title('Sarcastic Headlines Wordcloud')
    axs[0].axis('off')

    # Display word cloud for non-sarcastic headlines
    axs[1].imshow(wordcloud_non_sarcastic, interpolation='bilinear')
    axs[1].set_title('Non-Sarcastic Headlines Wordcloud')
    axs[1].axis('off')

    return fig

def plot_top_ngrams(text, n=2, top_k=10, title="Top N-Grams"):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_ngrams = words_freq[:top_k]

    # Create a new figure
    fig = plt.figure(figsize=(10, 6))
    x, y = map(list, zip(*top_ngrams))
    sns.barplot(x=y, y=x).set_title(title)
    return fig


def calculate_sentiment(text):
    # Ensure this function is properly implemented to perform sentiment analysis
    sentiments = [TextBlob(t).sentiment.polarity for t in text]
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiments, kde=True, color='blue')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt.gcf()

def plot_top_tfidf_terms_modified(text, top_k=10, title="Top TF-IDF Terms"):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text)
    feature_array = np.array(tfidf.get_feature_names_out())
    
    avg_scores = np.mean(tfidf_matrix, axis=0).A1
    top_indices = np.argsort(avg_scores)[::-1][:top_k]
    
    top_terms = feature_array[top_indices]
    top_scores = avg_scores[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scores, y=top_terms)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_flesch_reading_ease(df):
    # Calculating Flesch Reading Ease scores for each headline
    df['flesch_reading_ease'] = df['headline'].apply(textstat.flesch_reading_ease)

    # Create a figure for the plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Sarcastic readability
    sns.histplot(df[df['is_sarcastic'] == 1]['flesch_reading_ease'], kde=False, bins=20, color='red', ax=axs[0])
    axs[0].set_title('Flesch Reading Ease of Sarcastic Headlines')
    axs[0].set_xlabel('Flesch Reading Ease Score')
    axs[0].set_ylabel('Count')

    # Non-sarcastic readability
    sns.histplot(df[df['is_sarcastic'] == 0]['flesch_reading_ease'], kde=False, bins=20, color='blue', ax=axs[1])
    axs[1].set_title('Flesch Reading Ease of Non-Sarcastic Headlines')
    axs[1].set_xlabel('Flesch Reading Ease Score')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    return fig
