import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from surprise import Dataset, Reader, SVD
from textblob import TextBlob
import re
from html import unescape

# Initialize FastAPI app
app = FastAPI(title="Smart Product Recommender")

# Load dataset
df = pd.read_csv("data/Reviews.csv")
df = df[['UserId', 'ProductId', 'Score', 'Text']]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Filter frequent users/products
df = df[df['UserId'].map(df['UserId'].value_counts()) >= 10]
df = df[df['ProductId'].map(df['ProductId'].value_counts()) >= 10]

# Clean and score sentiment
def clean_review(text):
    text = unescape(text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

df['CleanText'] = df['Text'].apply(clean_review)
df['Sentiment'] = df['CleanText'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Build model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Valid users list
valid_users = df['UserId'].unique().tolist()

# Request schema
class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 5

# This endpoint must exist
@app.get("/users")
def list_users():
    print("/users endpoint hit!")
    return {"valid_users": valid_users}


# Main recommendation endpoint
@app.post("/recommend")
def recommend_products(req: RecommendationRequest):
    user_id = req.user_id
    top_n = req.top_n

    if user_id not in valid_users:
        return {"error": "User not found"}

    rated_products = df[df['UserId'] == user_id]['ProductId'].unique()
    all_products = df['ProductId'].unique()
    unrated_products = [pid for pid in all_products if pid not in rated_products]

    predictions = [(pid, model.predict(user_id, pid).est) for pid in unrated_products]
    top_n_preds = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []
    for pid, score in top_n_preds:
        review_row = df[df['ProductId'] == pid].iloc[0]
        recommendations.append({
            "ProductID": pid,
            "PredictedRating": round(score, 2),
            "SentimentScore": round(review_row['Sentiment'], 2),
            "SampleReview": review_row['CleanText'][:200]
        })

    return {
        "user": user_id,
        "recommendations": recommendations
    }
