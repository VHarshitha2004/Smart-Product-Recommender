import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Smart Product Recommender", layout="wide")
st.title("🛒 Smart Product Recommender")

st.write("Connecting to API at:", API_BASE)

@st.cache_data
def get_user_ids():
    try:
        response = requests.get(f"{API_BASE}/users")
        st.write("Status Code from /users:", response.status_code)
        return response.json()["valid_users"]
    except Exception as e:
        st.error(f"❌ Failed to load users: {e}")
        return []

user_ids = get_user_ids()

if user_ids:
    selected_user = st.selectbox("Choose a user ID", user_ids)
    top_n = st.slider("How many recommendations?", 1, 10, 5)

    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            payload = {"user_id": selected_user, "top_n": top_n}
            response = requests.post(f"{API_BASE}/recommend", json=payload)

            if response.status_code == 200:
                data = response.json()
                recs = data["recommendations"]
                st.success(f"Top {top_n} Recommendations for {selected_user}:")

                for rec in recs:
                    st.markdown(f"### 🛍️ Product: `{rec['ProductID']}`")
                    st.markdown(f"⭐ **Predicted Rating:** {rec['PredictedRating']}")
                    st.markdown(f"💬 **Sentiment Score:** {rec['SentimentScore']}")
                    st.markdown(f"📝 **Review Snippet:** {rec['SampleReview']}")
                    st.markdown("---")
            else:
                st.error("🚨 Failed to get recommendations.")
else:
    st.warning("No users available to recommend.")
