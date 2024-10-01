import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import re
import pickle

# Load the LSTM model and tokenizer
model = load_model('sentimentk.keras')
with open('tokenizerk.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to scrape all comments from YouTube
def get_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    while True:  # Loop to fetch all comments
        try:
            results = youtube.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText',
                maxResults=50, pageToken=next_page_token
            ).execute()

            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = results.get('nextPageToken')
            if next_page_token is None:  # Break the loop if no more comments
                break

        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            break

    return comments

# Function to analyze sentiments using the LSTM model
def analyze_sentiments(comments):
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    sentiment_labels = []

    sequences = tokenizer.texts_to_sequences(comments)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)

    for pred in predicted_classes:
        if pred == 0:
            sentiment_labels.append('Negative')
            sentiment_counts['negative'] += 1
        elif pred == 1:
            sentiment_labels.append('Neutral')
            sentiment_counts['neutral'] += 1
        else:
            sentiment_labels.append('Positive')
            sentiment_counts['positive'] += 1

    return sentiment_labels, sentiment_counts

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    pattern = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    return re.match(pattern, url) is not None

# Streamlit UI
st.title('YouTube Comment Sentiment Analyzer')

api_key = st.text_input('Enter your YouTube API Key:')
video_url = st.text_input('Enter YouTube Video URL:')

show_data = st.checkbox('Show fetched comments')

if st.button('Analyze'):
    if not is_valid_youtube_url(video_url):
        st.error("Invalid YouTube URL. Please enter a valid URL in the format: https://www.youtube.com/watch?v=VIDEO_ID")
    else:
        video_id = video_url.split('v=')[-1]  # Extract video ID
        if len(video_id) != 11:  # Ensure video ID length is correct
            st.error("Invalid video ID. Please check the URL.")
        else:
            st.info(f'Fetching all comments for the video...')
            
            comments = get_comments(video_id, api_key)
            st.success(f"Fetched {len(comments)} comments.")
            
            # Exclude the first comment
            filtered_comments = comments[1:]  # Skip the first comment

            if show_data:
                st.write("### Fetched Comments:")
                comments_df = pd.DataFrame({'Comment': filtered_comments})
                st.dataframe(comments_df, height=300)  # Display comments if checkbox is checked

            # Analyze only the filtered comments
            sentiment_labels, sentiment_counts = analyze_sentiments(filtered_comments)

            st.write("### Filtered Comments and Sentiments:")
            comments_df = pd.DataFrame({
                'Comment': filtered_comments,
                'Sentiment': sentiment_labels
            })

            # Display the DataFrame in a scrollable format
            st.dataframe(comments_df, height=300)  # Adjust height as needed

            total_comments = sum(sentiment_counts.values())
            sentiment_percentages = {k: (v / total_comments) * 100 for k, v in sentiment_counts.items()}

            # Display bar chart of sentiment percentages
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(sentiment_percentages.keys(), sentiment_percentages.values(), color=['red', 'blue', 'green'])
            ax.set_ylabel('Percentage')
            ax.set_title('Sentiment Analysis of Comments')

            # Add text labels above the bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{int(yval)}", ha='center', va='bottom')

            st.pyplot(fig)

            # Display pie chart of sentiment percentages
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie(sentiment_percentages.values(), labels=sentiment_percentages.keys(), autopct='%1.1f%%', colors=['red', 'blue', 'green'])
            ax2.set_title('Sentiment Distribution')
            st.pyplot(fig2)

