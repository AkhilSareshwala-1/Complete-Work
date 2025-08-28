import os
import json
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from kiteconnect import KiteConnect
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LangChain & FAISS imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# ---------------------------------------------
# Logging & Flask setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tradevision")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.urandom(24)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# -------------------------
# Gemini / LLM config
GEMINI_API_KEY = "AIzaSyAU_7XCPFKWQKU_OKvt-XlgBZ3HmESESIc"
GEMINI_MODEL_CANDIDATES = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
_chat_model_cache = {"model": None, "client": None}

def _make_chat_client(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=model_name, temperature=0.2)
 
def _select_working_client():
    if _chat_model_cache["client"] is not None:
        return _chat_model_cache["client"], _chat_model_cache["model"]

    probe_prompt = "Reply with a single word: ok"
    last_error = None
    for candidate in GEMINI_MODEL_CANDIDATES:
        try:
            client = _make_chat_client(candidate)
            resp = client.invoke(probe_prompt)
            text = getattr(resp, "content", None) or str(resp)
            if "ok" in text.lower():
                _chat_model_cache.update({"client": client, "model": candidate})
                logger.info(f"[LLM] using model: {candidate}")
                return client, candidate
        except Exception as e:
            last_error = e
            logger.debug(f"[LLM] probe failed for {candidate}: {e}")
    raise RuntimeError(f"No working Gemini model found. Last error: {last_error}")

# ---------------------------------------------------------------------------------
# FAISS setup for IPO and Delivery
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ipo_faiss, delivery_faiss = None, None

def load_faiss_index():
    global ipo_faiss, delivery_faiss
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # IPO
    ipo_docs = []
    try:
        with open("ipo_analysis.txt", encoding="utf-8") as f:
            text = f.read().strip()
        blocks = [blk for blk in text.split("\n\n") if blk.strip()]
        ipo_docs = [Document(page_content=blk) for blk in blocks]
    except Exception as e:
        logger.error(f"[FAISS] IPO load error: {e}")

    # Delivery
    delivery_docs = []
    try:
        with open("delivery_suggestion.txt", encoding="utf-8") as f:
            text = f.read().strip()
        blocks = [blk for blk in text.split("```") if blk.strip()]
        delivery_docs = [Document(page_content=blk) for blk in blocks]
    except Exception as e:
        logger.error(f"[FAISS] Delivery load error: {e}")

    if ipo_docs:
        ipo_faiss = FAISS.from_documents(splitter.split_documents(ipo_docs), embedding_model)
        logger.info(f"[FAISS] IPO index built with {len(ipo_docs)} documents")
    if delivery_docs:
        delivery_faiss = FAISS.from_documents(splitter.split_documents(delivery_docs), embedding_model)
        logger.info(f"[FAISS] Delivery index built with {len(delivery_docs)} documents")

def classify_query(prompt: str) -> str:
    low = prompt.lower()
    if any(k in low for k in ["ipo", "listing", "gmp", "issue price"]):
        return "ipo"
    if any(k in low for k in ["delivery", "holding", "stock pick", "long term"]):
        return "delivery"
    return "llm"

# -------------------------
# Chatbot (Hybrid FAISS + LLM)
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json(force=True) or {}
        prompt = (data.get("question") or "").strip()
        if not prompt:
            return jsonify({"response": "Please enter your question."})

        qtype = classify_query(prompt)
        if qtype == "ipo" and ipo_faiss:
            docs = ipo_faiss.similarity_search(prompt, k=3)
            resp = "\n\n---\n\n".join(doc.page_content for doc in docs)
            return jsonify({"response": f"**IPO Insights:**\n\n{resp}", "model": "FAISS-IPO"})
        if qtype == "delivery" and delivery_faiss:
            docs = delivery_faiss.similarity_search(prompt, k=3)
            resp = "\n\n---\n\n".join(doc.page_content for doc in docs)
            return jsonify({"response": f"**Delivery Suggestions:**\n\n{resp}", "model": "FAISS-Delivery"})

        # Fallback to Gemini LLM
        client, model_used = _select_working_client()
        system_prompt = (
            "You are TradeVision, a highly skilled trading and investment assistant. "
            "Your role is to provide accurate, concise, and insightful answers on stocks, IPOs, "
            "market analysis, portfolio strategies, and risk management. "
            f"\n\nUser question: {prompt}"
        )
        ans = client.invoke(system_prompt)
        text = getattr(ans, "content", None) or str(ans)                                                            
        return jsonify({"response": text, "model": model_used})

    except Exception as e:
        logger.exception("Unexpected /chatbot error")
        return jsonify({"response": f"Error: {str(e)}"}), 500

# -------------------------
# Kite Helpers
API_KEY = os.getenv("KITE_API_KEY", "fispy0ikjo710gqt")
ACCESS_TOKEN_FILE = os.getenv("ACCESS_TOKEN_FILE", "access_token.txt")

def get_kite_client():
    try:
        with open(ACCESS_TOKEN_FILE, "r") as f:
            access_token = f.read().strip()
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        return kite
    except Exception as e:
        logger.error(f"Failed to initialize KiteConnect: {e}")
        return None

def format_currency(value):
    try:
        return "{:,.2f}".format(float(value))
    except Exception:
        return "0.00"

def calculate_pnl(position):
    try:
        if position.get('product') in ['BO', 'CO']:
            return (position.get('sell_price', 0) - position.get('buy_price', 0)) * position.get('quantity', 0)
        return (position.get('last_price', 0) - position.get('buy_price', 0)) * position.get('quantity', 0)
    except Exception:
        return 0.0

# -------------------------
# Portfolio routes (login/dashboard/logout)
@app.route("/portfolio/login", methods=['GET', 'POST'])
def portfolio_login():
    if request.method == 'POST': 
        api_key = request.form.get('api_key')
        access_token = request.form.get('access_token')
        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            profile = kite.profile()
            session['api_key'] = api_key
            session['access_token'] = access_token
            session['user_name'] = profile.get('user_name', 'Unknown User')
            return redirect(url_for('portfolio_dashboard'))
        except Exception:
            return render_template("portfolio_login.html", error="Invalid API key or access token.")
    return render_template("portfolio_login.html")

@app.route("/portfolio/dashboard")
def portfolio_dashboard():
    if 'api_key' not in session or 'access_token' not in session:
        return redirect(url_for('portfolio_login'))
    try:
        kite = KiteConnect(api_key=session['api_key'])
        kite.set_access_token(session['access_token'])                                                                                  
        profile = kite.profile()
        margins = kite.margins()
        holdings = kite.holdings()
        positions = kite.positions()
        orders = kite.orders()
        trades = kite.trades()
        for position in positions.get('net', []):
            position['pnl'] = calculate_pnl(position)
        summary = {
            'total_value': sum(float(h.get('last_price', 0)) * float(h.get('quantity', 0)) for h in holdings),              
            'total_equity': margins.get('equity', {}).get('net', 0),
            'available_margin': margins.get('equity', {}).get('available', {}).get('live_balance', 0),
            'total_pnl': sum(float(p.get('pnl', 0)) for p in positions.get('net', []))
        }
        return render_template("portfolio_dashboard.html",
                               profile=profile, margins=margins, holdings=holdings,
                               positions=positions, orders=orders, trades=trades,
                               summary=summary, format_currency=format_currency,
                               now=datetime.now().strftime('%d %b %Y %H:%M:%S'))
    except Exception:
        return render_template("portfolio_login.html", error="Session expired. Please login again.")

@app.route("/portfolio/logout")
def portfolio_logout():
    session.clear()
    return redirect(url_for('portfolio_login'))

# -------------------------
# IPO analyzer (file-based) & Delivery suggestions
@app.route("/ipo-analyzer")
def ipo_analyzer():
    ipo_entries = []
    try:
        with open("ipo_analysis.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        current = {}
        for line in lines:
            if not line.strip():
                continue 
            if "\t" in line:
                if current:
                    ipo_entries.append(current)
                parts = line.split("\t", 1)
                current = {"timestamp": parts[0], "suggestion": parts[1], "reasoning": ""}
            else:
                current["reasoning"] += line + "\n"
        if current:                                                                                                 
            ipo_entries.append(current)
    except Exception as e:
        ipo_entries = [{"timestamp": "N/A", "suggestion": "No data available", "reasoning": str(e)}]
    return render_template("ipo_analyzer.html", ipo_entries=ipo_entries)

@app.route("/delivery-suggestions")
def delivery_suggestions():
    suggestions = []
    current = {}
    try:
        with open("delivery_suggestion.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            line = line.strip()  
            if not line:
                continue
            if line.endswith("```"):
                if current and 'name' in current and 'json' in current:
                    try:
                        current['parsed'] = json.loads(current['json'])
                    except Exception:
                        current['parsed'] = {"error": "Invalid JSON"}
                    suggestions.append(current) 
                    current = {}
            elif not current.get('name') and line:
                current['name'] = line.split()[0]
            elif line.startswith("```"):
                continue
            else:
                current['json'] = current.get('json', '') + line
    except Exception as e:
        logger.error(f"Failed to read delivery suggestions: {e}")
    return render_template("delivery_suggestions.html", suggestions=suggestions)

# -------------------------
# Backtester
portfolio_csv = """Company Symbol,Company Name,Buy Price,Sell Price,Stop Loss,Holding Period (Days),Quantity  
RELIANCE,Reliance Industries Ltd,2840,2950,2780,5,30
TCS,TCS Ltd,3770,3900,3700,6,20
HDFCBANK,HDFC Bank Ltd,1675,1745,1640,4,40
INFY,Infosys Ltd,1520,1600,1480,7,50
ICICIBANK,ICICI Bank Ltd,790,830,770,6,35
"""
portfolio_df = pd.read_csv(StringIO(portfolio_csv))

@app.route('/backtest', methods=['GET', 'POST'])
def backtester():
    date = None
    results = []
    analysis = {}
    portfolio = portfolio_df.to_dict(orient='records')
    if request.method == 'POST':
        date = request.form.get('date')
        if date:
            try:
                kite = get_kite_client()
                if kite is None:
                    raise Exception("Kite client init failed")
                from_date = datetime.strptime(date, "%Y-%m-%d")
                to_date = from_date + timedelta(days=1)
                for row in portfolio:
                    symbol = row["Company Symbol"]
                    tradingsymbol = f"NSE:{symbol}"
                    try:
                        ltp_data = kite.ltp([tradingsymbol])
                        instrument_token = ltp_data.get(tradingsymbol, {}).get("instrument_token")
                        if instrument_token is None:
                            raise Exception("instrument_token missing")
                        ohlc = kite.historical_data(instrument_token, from_date, to_date, interval="minute") 
                        df = pd.DataFrame(ohlc)
                        actual_low = df["low"].min() if not df.empty else None
                        actual_high = df["high"].max() if not df.empty else None
                        if actual_high and actual_high >= row["Sell Price"]:
                            status = "Target Hit"
                        elif actual_low and actual_low <= row["Stop Loss"]:
                            status = "Stop Loss Hit"
                        else:
                            status = "None Hit"
                    except Exception as e:
                        actual_low = actual_high = None
                        status = f"Data fetch failed: {e}"
                    results.append({
                        "Company": row['Company Name'],
                        "Buy_Price": row['Buy Price'],
                        "Sell_Price": row['Sell Price'],
                        "Stop_Loss": row['Stop Loss'],
                        "Low": round(actual_low, 2) if actual_low else None,
                        "High": round(actual_high, 2) if actual_high else None,   
                        "Status": status
                    })
                analysis = {
                    "companies": json.dumps([r["Company"] for r in results]),
                    "highs": json.dumps([r["High"] for r in results]),
                    "lows": json.dumps([r["Low"] for r in results]),
                    "buy_prices": json.dumps([r["Buy_Price"] for r in results]),
                    "sell_prices": json.dumps([r["Sell_Price"] for r in results])
                }
            except Exception as e:
                results = [{"Company": "ERROR", "Status": str(e)}]
    return render_template('backtester.html', portfolio=portfolio, date=date, results=results, analysis=analysis)

#x ####x####x#x#x#x#x#x#x#x
#x xHome
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------
# Run
if __name__ == "__main__":
    load_faiss_index()
    app.run(debug=True, host="0.0.0.0", port=5000)