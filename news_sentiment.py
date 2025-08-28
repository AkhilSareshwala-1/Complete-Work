import os
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage

# Configuration
os.environ["GOOGLE_API_KEY"] = "AIzaSyC7R_1is8Q0VETmrWDMji8NWE02Yj0czRE"
RSS_FEEDS = {
    # 🏦 Indian Markets & Economy
    "Moneycontrol – Stocks": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Economic Times – Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Livemint – Markets": "https://www.livemint.com/rss/markets",
    "Trade Brains – Investing": "https://tradebrains.in/feed/",

    # 🌍 Global Markets & Companies (New Verified Additions)
    "Wall Street Journal - Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Financial Times - Global Companies": "https://www.ft.com/companies?format=rss",
    "MarketBeat - Earnings Reports": "https://www.marketbeat.com/feed/",
    "CNBC - Market Movers": "https://www.cnbc.com/id/19746125/device/rss/rss.html",
    "SeekingAlpha - Stocks": "https://seekingalpha.com/market_currents.xml",
    "Yahoo Finance - Market News": "https://finance.yahoo.com/news/rssindex",
    "Investing.com - Stocks": "https://www.investing.com/rss/news.rss",
    "Business Insider - Markets": "https://markets.businessinsider.com/rss",

    # ⚖️ Legal & Regulatory
    "Bar & Bench – Legal News": "https://barandbench.com/feed/",
    "iPleaders – Business Law": "https://blog.ipleaders.in/feed/",
    "SpicyIP – Intellectual Property": "https://spicyip.com/feed/",

    # ⚡ Energy & Oil
    "OilPrice.com – Crude Oil": "https://oilprice.com/rss/main",

    # 🏦 Banking & Finance
    "ET BFSI News": "https://bfsi.economictimes.indiatimes.com/rss/topstories",
    "Livemint – Money": "https://www.livemint.com/rss/money",

    # 📈 Market Strategy & Education
    "Capitalmind – Market Insights": "https://www.capitalmind.in/feed/",
    "Finshots": "https://finshots.in/rss/",

    # 🚗 Auto
    "ET Auto – Industry News": "https://auto.economictimes.indiatimes.com/rss/industry",

    # 🏗️ Infrastructure & Real Estate
    "ET Realty – Infra & Projects": "https://realty.economictimes.indiatimes.com/rss/infrastructure",

    # 🌾 Agriculture & Commodities
    "Business Line – Agri Business": "https://www.thehindubusinessline.com/economy/agri-business/feeder/default.rss",

    # 💻 Tech & IT
    "Livemint – Technology": "https://www.livemint.com/rss/technology",
    "Inc42 – Indian Startups": "https://inc42.com/feed/",

    # � Pharmaceuticals & Healthcare
    "ET Healthworld – Pharma": "https://health.economictimes.indiatimes.com/rss/pharma"
}

# Updated Category Mapping
SOURCE_TO_CATEGORY = {
    # Indian Markets & Economy
    "Moneycontrol – Stocks": "Indian Markets & Economy",
    "Economic Times – Markets": "Indian Markets & Economy",
    "Livemint – Markets": "Indian Markets & Economy",
    "Trade Brains – Investing": "Indian Markets & Economy",
    
    # Global Markets & Companies
    "Wall Street Journal - Markets": "Global Markets & Companies",
    "Financial Times - Global Companies": "Global Markets & Companies", 
    "MarketBeat - Earnings Reports": "Global Markets & Companies",
    "CNBC - Market Movers": "Global Markets & Companies",
    "SeekingAlpha - Stocks": "Global Markets & Companies",
    "Yahoo Finance - Market News": "Global Markets & Companies",
    "Investing.com - Stocks": "Global Markets & Companies",
    "Business Insider - Markets": "Global Markets & Companies",
    
    # Legal & Regulatory
    "Bar & Bench – Legal News": "Legal & Regulatory",
    "iPleaders – Business Law": "Legal & Regulatory",
    "SpicyIP – Intellectual Property": "Legal & Regulatory",
    
    # Energy & Oil
    "OilPrice.com – Crude Oil": "Energy & Oil",
    
    # Banking & Finance
    "ET BFSI News": "Banking & Finance",
    "Livemint – Money": "Banking & Finance",
    
    # Market Strategy & Education
    "Capitalmind – Market Insights": "Market Strategy & Education",
    "Finshots": "Market Strategy & Education",
    
    # Auto
    "ET Auto – Industry News": "Auto",
    
    # Infrastructure & Real Estate
    "ET Realty – Infra & Projects": "Infrastructure & Real Estate",
    
    # Agriculture & Commodities
    "Business Line – Agri Business": "Agriculture & Commodities",
    
    # Tech & IT
    "Livemint – Technology": "Tech & IT",
    "Inc42 – Indian Startups": "Tech & IT",
    
    # Pharmaceuticals & Healthcare
    "ET Healthworld – Pharma": "Pharmaceuticals & Healthcare"
}

def fetch_last_2hours_news(feed_url):
    """Fetch news from last 2 hours"""
    news_entries = []
    feed = feedparser.parse(feed_url)
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    time_2hours_ago = now - timedelta(hours=2)
    
    for entry in feed.entries:
        published_time = None
        if "published_parsed" in entry:
            published_time = datetime(*entry.published_parsed[:6]) + timedelta(hours=5, minutes=30)
        elif "published" in entry:
            try:
                published_time = datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %Z")
                if "GMT" in entry.published:
                    published_time += timedelta(hours=5, minutes=30)
            except ValueError:
                continue
        
        if not published_time or not (time_2hours_ago <= published_time <= now):
            continue
            
        news_entries.append({
            "Title": entry.title,
            "Content": f"{entry.title}. {entry.get('description', '')}",
            "Category": SOURCE_TO_CATEGORY.get(feed.feed.get("title", "Unknown"), "Uncategorized")
        })
    return news_entries

def get_latest_news():
    """Get all news from RSS feeds"""
    if os.path.exists("news_data.csv"):
        file_time = datetime.fromtimestamp(os.path.getmtime("news_data.csv"))
        if (datetime.now() - file_time) < timedelta(hours=2):
            return pd.read_csv("news_data.csv")
    
    all_news = []
    for source, url in RSS_FEEDS.items():
        try:
            all_news.extend(fetch_last_2hours_news(url))
        except Exception as e:
            print(f"⚠️ Error fetching {source}: {e}")
    
    df = pd.DataFrame(all_news) if all_news else pd.DataFrame()
    if not df.empty:
        df.to_csv("news_data.csv", index=False)
    return df

def setup_search():
    """Initialize vector store and LLM"""
    news_df = get_latest_news()
    if news_df.empty:
        print("❌ No news articles available")
        return None, None
    
    documents = [
        {"page_content": row['Content'], "metadata": {
            "title": row['Title'],
            "category": row['Category']
        }} for _, row in news_df.iterrows()
    ]
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(
        [doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    return vectorstore, llm

def analyze_news(vectorstore, llm, query):
    """Generate comprehensive analysis with stock prediction"""
    relevant_docs = vectorstore.similarity_search(query, k=15)  # Get more articles for better analysis
    
    if not relevant_docs:
        print(f"No recent news found about {query}")
        return
    
    # Prepare context
    context = "\n\n".join([
        f"Headline {i+1}: {doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    ])
    
    prompt = f"""Analyze all recent news about {query} and provide:
    
    1. EXECUTIVE SUMMARY (3-4 sentences):
    - Key developments and why they matter
    - Major players involved
    
    2. SENTIMENT ANALYSIS:
    - Overall sentiment (Positive/Negative/Neutral)
    - Sentiment strength (1-10 scale)
    - Key positive/negative factors
    
    3. STOCK MOVEMENT PREDICTION:
    - % probability of price increase (0-100%)
    - % probability of price decrease (0-100%) 
    - % probability of no significant change
    - Confidence level in prediction (High/Medium/Low)
    - Key factors influencing prediction
    
    News Articles:
    {context}"""
    
    print(f"\n📊 Generating comprehensive analysis for {query}...")
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n" + "="*60)
    print(f"🔍 ANALYSIS REPORT: {query.upper()}")
    print("="*60)
    print(response.content)
    print("="*60)

def main():
    print("📡 Loading financial news data...")
    vectorstore, llm = setup_search()
    if not vectorstore:
        return
    
    while True:
        print("\n" + "="*60)
        query = input("\nEnter company/stock to analyze (or 'quit'): ").strip()
        if query.lower() in ('quit', 'exit'):
            break
            
        if query:
            analyze_news(vectorstore, llm, query)

if __name__ == '__main__':
    main()