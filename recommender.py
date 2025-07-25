import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from surprise import Dataset, Reader, SVD
from textblob import TextBlob
import re
from html import unescape

# Initialize FastAPI app
app = FastAPI(title="Smart Product Recommender API")

# Load and prepare data
df = pd.read_csv("data/Reviews.csv")
df = df[['UserId', 'ProductId', 'Score', 'Text']]
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Filter users/products with enough reviews
df = df[df['UserId'].map(df['UserId'].value_counts()) >= 10]
df = df[df['ProductId'].map(df['ProductId'].value_counts()) >= 10]

# Clean text
def clean_review(text):
    text = unescape(text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

df['CleanText'] = df['Text'].apply(clean_review)
df['Sentiment'] = df['CleanText'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Build SVD model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# ✅ Collect valid users once
valid_user_ids = df['UserId'].unique().tolist()

# Define request model
class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 5

# API endpoint to get recommendations
@app.post("/recommend")
def recommend_products(req: RecommendationRequest):
    user_id = req.user_id
    top_n = req.top_n

    if user_id not in valid_user_ids:
        return {"error": f"User '{user_id}' not found. Try one from /users endpoint."}

    all_products = df['ProductId'].unique()
    rated = df[df['UserId'] == user_id]['ProductId'].unique()
    unrated = [pid for pid in all_products if pid not in rated]

    predictions = [(pid, model.predict(user_id, pid).est) for pid in unrated]
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    result = []
    for pid, score in top_predictions:
        review_row = df[df['ProductId'] == pid].iloc[0]
        result.append({
            "ProductID": pid,
            "PredictedRating": round(score, 2),
            "SentimentScore": round(review_row['Sentiment'], 2),
            "SampleReview": review_row['CleanText'][:200]
        })

    return {
        "user_id": user_id,
        "recommendations": result
    }

# ✅ Endpoint to return valid user IDs
@app.get("/users")
def list_users():
    return {"valid_users": valid_user_ids[:20]}  # Show first 20 to test
