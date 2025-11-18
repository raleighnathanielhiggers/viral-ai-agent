import requests  # For X API calls
import pandas as pd  # For organizing data like a spreadsheet
from groq import Groq  # For AI analysis
import os  # For loading keys securely

# Load your keys (replace paths if needed)

BEARER_TOKEN = os.getenv('X_BEARER_TOKEN')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Set up Groq AI
client = Groq(api_key=GROQ_API_KEY)

def get_viral_posts():
    query = "(min_faves:1000 -min_faves:10000) -is:retweet lang:en"  # Viral-ish, English, no RTs
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": query,
        "tweet.fields": "public_metrics,author_id,created_at,text",
        "expansions": "author_id",
        "user.fields": "public_metrics,username",
        "max_results": 100  # Pull 100, filter later
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("Error:", response.text)
        return pd.DataFrame()
    
    data = response.json()
    tweets = []
    users = {u['id']: u for u in data.get('includes', {}).get('users', [])}
    for t in data.get('data', []):
        user = users.get(t['author_id'], {})
        if user.get('public_metrics', {}).get('followers_count', 0) < 10000:  # Small account
            tweets.append({
                'user': user.get('username'),
                'followers': user.get('public_metrics', {}).get('followers_count', 0),
                'text': t['text'],
                'likes': t['public_metrics']['like_count'],
                'created_at': t['created_at']
            })
    return pd.DataFrame(tweets)

def analyze_why_viral(text):
    prompt = f"""
    Analyze this tweet from a small account: "{text[:500]}..."
    Explain in 3 short bullets why it likely went viral:
    - Emotional trigger or hook?
    - Timing, trend, or relatability?
    - Format, emojis, or surprise element?
    """
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=150
    )
    return completion.choices[0].message.content

# Run the scan
df = get_viral_posts()
if not df.empty:
    df['analysis'] = df['text'].apply(analyze_why_viral)
    df.to_csv('viral_results.csv', index=False)  # Save to file
    print(df[['user', 'likes', 'text', 'analysis']].head(10))  # Show top 10
else:
    print("No viral posts found or API error.")