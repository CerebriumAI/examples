import json
import requests
import re
from dataclasses import dataclass
from analyst import BetAnalyst

@dataclass
class MarketData:
    question: str
    yes_price: str
    no_price: str

def _fetch_api_data(url: str):
    try:
        res = requests.get(url)
        res.raise_for_status()
        obj = res.json()
        return obj
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching Kalshi market data: {e}")
        
def get_kalshi_market(ticker: str) -> MarketData:
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
    raw_data = _fetch_api_data(url)

    try:
        market = raw_data['market']
        return MarketData(
            question = market['title'],
            yes_price=float(market['yes_ask']),
            no_price=float(market['no_ask'])
        )
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Error parsing Kalshi data structure: {e}") from e

def get_polymarket_market(slug: str) -> MarketData:
    url = f"https://gamma-api.polymarket.com/markets/slug/{slug}" # slug
    raw_data = _fetch_api_data(url)
    
    try:
        poly_values = json.loads(raw_data['outcomePrices'])
        yes_price, no_price = [float(v) for v in poly_values]

        return MarketData(
            question = raw_data['question'],
            yes_price=yes_price,
            no_price=no_price
        )
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"Error parsing Kalshi data structure: {e}") from e
        

def getMarket(is_kalshi, ticker):
    if is_kalshi: 
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}" # market ticker
    else:
        url = f"https://gamma-api.polymarket.com/markets/slug/{ticker}" # slug
    try:
        res = requests.get(url)
        res.raise_for_status()
        obj = res.json()
        return obj
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching Kalshi market data: {e}")


def evaluate(analyst, question):
    # Generate questions using OpenAI API
    relevant_questions = analyst.get_relevant_questions(question)
    # Use Exa semantic search to retrieve answers to questions
    answers = analyst.get_web_info(relevant_questions)

    information = ""
    for i, v in enumerate(relevant_questions):
        information += f"INFORMATION {i+1}: \n"
        information += f"QUESTION {i+1}: {v}\n"
        information += f"ANSWER {i+1}: {answers[i]} \n\n"

    information.rstrip("\n")
    
    # Passes relevant Q&As to OpenAI API and generates Y/N percentage with explanation
    yes, no, explanation = analyst.get_binary_answer_with_percentage(information, question)
    return yes, no, explanation

def predict(kalshi_ticker, poly_slug):
    kalshi_market = get_kalshi_market(kalshi_ticker)
    poly_market = get_polymarket_market(poly_slug)
    question = poly_market.question # we use polymarket because they have direct question

    print(f"Question: {question}") 
    
    analyst = BetAnalyst()    
    pred_yes, pred_no, explanation = evaluate(analyst, question)

    match_yes = re.search(r"(\d+)%", pred_yes)
    match_no = re.search(r"(\d+)%", pred_no)
    pred_yes = float(match_yes.group(1))
    pred_no = float(match_no.group(1))
    
    kalshi_buy_yes = kalshi_market.yes_price < pred_yes
    kalshi_buy_no = kalshi_market.no_price < pred_no

    poly_buy_yes = poly_market.yes_price < pred_yes
    poly_buy_no = poly_market.no_price < pred_no

    return {
        "kalshi": {
            "buy_yes":kalshi_buy_yes,
            "buy_no": kalshi_buy_no,
            "edge": max(pred_yes-kalshi_market.yes_price, pred_no-kalshi_market.no_price),   
        },
        "polymarket": {
            "buy_yes":poly_buy_yes,
            "buy_no": poly_buy_no,
            "edge": max(pred_yes-poly_market.yes_price, pred_no-poly_market.no_price),
        },
        "yes": pred_yes,
        "no": pred_no,
        "explanation": explanation
    }
