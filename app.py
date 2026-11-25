import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
model = joblib.load('zomato_sentiment_model.pkl')
tfidf = joblib.load('zomato_tfidf_vectorizer.pkl')

# Load merged dataset
df = pd.read_csv("zomato_merged_dataset.csv")

stop_words = set(stopwords.words('english'))

for word in ['not', 'no', 'nor', 'never', 'dont', 'doesnt',
              'didnt', 'isnt', 'cant', 'couldnt', 'wouldnt',
              'shouldnt', 'wont']:
    stop_words.discard(word)

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )
    text = ' '.join(stemmer.stem(word) for word in text.split())
    return text

st.set_page_config(
    page_title="Zomato Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #fff8f5;
}
h1 {
    color: #e23744;
}
</style>
""", unsafe_allow_html=True)

st.header("📊 Dataset Overview")

total_reviews = len(df)

positive_reviews = len(df[df['Sentiment'] == 'Positive'])
negative_reviews = len(df[df['Sentiment'] == 'Negative'])

col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews", positive_reviews)
col3.metric("Negative Reviews", negative_reviews)
st.subheader("Sentiment Distribution")

sentiment_counts = df['Sentiment'].value_counts()

st.bar_chart(sentiment_counts)

st.subheader("📋 Sample Reviews from Dataset")

st.dataframe(
    df[['Restaurant', 'Review', 'Rating', 'Sentiment']].head(10),
    use_container_width=True
)
st.subheader("🔎 Filter Reviews by Sentiment")

selected_sentiment = st.selectbox(
    "Select Sentiment",
    ["All", "Positive", "Negative"]
)

if selected_sentiment == "All":
    filtered_df = df[['Restaurant', 'Review', 'Rating', 'Sentiment']]
else:
    filtered_df = df[df['Sentiment'] == selected_sentiment][
        ['Restaurant', 'Review', 'Rating', 'Sentiment']
    ]

st.dataframe(filtered_df.head(50), use_container_width=True)
st.title("🍽️ Zomato Sentiment Analyzer")
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    "### Understand restaurant reviews using Machine Learning"
)

st.write(
    "Enter a restaurant review below and let our ML model "
    "predict whether the review expresses Positive or Negative sentiment."
)

st.divider()

review = st.text_area(
    "✍️ Write your restaurant review",
    placeholder="Example: The food was delicious and the service was excellent...",
    height=150
)

if st.button("🔍 Analyze Review", use_container_width=True):

    if review.strip():

        cleaned_review = preprocess_text(review)
        vectorized_review = tfidf.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]

        probabilities = model.predict_proba(vectorized_review)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100
        st.session_state.history.append({
    "Review": review,
    "Sentiment": prediction,
    "Confidence": f"{confidence:.2f}%"
})

        st.divider()

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")
            st.write("The review expresses an overall positive sentiment.")
        else:
            st.error("😞 Negative Sentiment")
            st.write("The review expresses an overall negative sentiment.")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

    else:
        st.warning("Please enter a review before analyzing.")

st.divider()

st.markdown("### 💡 Try Sample Reviews")

st.markdown("""
- The food was delicious and the staff was very friendly.
- The food was terrible and the service was very bad.
- The restaurant had a wonderful ambience and tasty food.
""")

st.caption("Mini Project | Restaurant Review Sentiment Analysis")
if st.session_state.history:

    st.divider()

    st.subheader("📊 Review Analysis History")

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()