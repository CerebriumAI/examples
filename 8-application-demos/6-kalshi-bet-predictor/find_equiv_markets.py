import csv
import os
import requests
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# --- Config ---
SIMILARITY_THRESHOLD = 0.70 # threshold for cosine simlarity
MAX_MARKET_LIMIT = 40000 # max number of active & open markets to gather
TOP_K = 5  # number of top Polymarket markets to check for each Kalshi market
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
POLYMARKET_API_URL = "https://clob.polymarket.com/markets"
OUTPUT_FILE = "markets.csv"

def get_kalshi_markets() -> List[Dict[str, Any]]:
    print("Fetching Kalshi markets...")
    markets_list = []
    cursor = ""
    try:
        while True:
            params = {'limit': 1000}
            if cursor:
                params['cursor'] = cursor

            response = requests.get(KALSHI_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if 'markets' not in data:
                print("Error: 'markets' key not in Kalshi response.")
                break

            for market in data['markets']:
                if market['status'] == 'active' and market['market_type'] == 'binary':

                    markets_list.append({
                        'platform': 'Kalshi',
                        'title': market['title'],
                        'ticker': market['ticker'],
                        'url': f"https://kalshi.com/markets/{market['ticker']}",
                        'event_url': f"https://kalshi.com/markets/{market['event_ticker']}",
                        'close_date': market['close_time']
                    })

            cursor = data['cursor']
            print(f"Found {len(markets_list)} active and open markets")

            if len(markets_list) > MAX_MARKET_LIMIT or not cursor:
                break

        print(f"Found {len(markets_list)} open binary markets on Kalshi.")
        return markets_list

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Kalshi markets: {e}")
        return []
    
def get_kalshi_market(ticker):
    title = requests.get(f"{KALSHI_API_URL}/{ticker}")
    title = title.json()
    return title['market']['title']

def get_polymarket_markets() -> List[Dict[str, Any]]:
    print("Fetching Polymarket markets...")
    markets_list = []
    next_cursor = None

    try:
        while True:
            params = {}
            if next_cursor:
                params['next_cursor'] = next_cursor

            response = requests.get(POLYMARKET_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            market_list_page = data['data']
            if not market_list_page:
                break

            for market in market_list_page:
                if market.get('active') and not market.get('closed'):
                    markets_list.append({
                        'platform': 'Polymarket',
                        'title': market.get('question'),
                        'id': market.get('condition_id'),
                        'url': f"https://polymarket.com/event/{market.get('market_slug')}",
                        'close_date': market.get('end_date_iso')
                    })

            next_cursor = data.get('next_cursor')
            print(f"Found {len(markets_list)} active and open markets")

            if len(markets_list) > MAX_MARKET_LIMIT or not next_cursor or next_cursor == 'LTE=':
                break

        print(f"Found {len(markets_list)} open markets on Polymarket.")
        return markets_list

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Polymarket markets: {e}")
        return []


def find_similar_markets(kalshi_markets, poly_markets, threshold=0.9, top_k=TOP_K):
    print("\nLoading NLP model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    kalshi_titles = [m['title'] for m in kalshi_markets]
    poly_titles = [m['title'] for m in poly_markets]

    if not kalshi_titles or not poly_titles:
        print("Not enough market data to compare.")
        return []

    print("Encoding titles into embeddings...")
    kalshi_embeddings = model.encode(kalshi_titles, convert_to_numpy=True, normalize_embeddings=True)
    poly_embeddings = model.encode(poly_titles, convert_to_numpy=True, normalize_embeddings=True)

    print(f"Building vector index for {len(poly_embeddings)} Polymarket markets...")
    dim = poly_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(poly_embeddings)

    print(f"Querying top {top_k} nearest Polymarket markets for each Kalshi market...")
    scores, indices = index.search(kalshi_embeddings, top_k)

    potential_matches = []
    for i, kalshi_market in enumerate(kalshi_markets):
        for j in range(top_k):
            score = float(scores[i][j])
            if score >= threshold:
                poly_market = poly_markets[indices[i][j]]
                potential_matches.append({
                    'score': score,
                    'kalshi_market': kalshi_market,
                    'poly_market': poly_market
                })
        if i % 100 == 0:
            print(f"Processed {i}/{len(kalshi_markets)} Kalshi markets...")


    return potential_matches
    
def interactive_save(matches: List[Dict[str, Any]]):
    print("\n--- Review Mode ---")
    print("Press 'y' to save a match, anything else to skip.\n")
    
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["kalshi_ticker", "poly_slug"])

        for i, match in enumerate(matches):
            kalshi_ticker = match['kalshi_market']['ticker']
            poly_slug = match['poly_market']['url'].split("event/")[1]
            kalshi_title = get_kalshi_market(kalshi_ticker)
            poly_title = match['poly_market']['title']
            score = match['score']

            print(f"\nMatch #{i+1} (Score: {score:.4f})")
            print(f"[KALSHI]     {kalshi_title}")
            print(f"[POLYMARKET] {poly_title}")
            print(f"  > Kalshi URL:    {match['kalshi_market']['url']}")
            print(f"  > Polymarket URL:{match['poly_market']['url']}")

            choice = input("Save this match? (y/n): ").strip().lower()
            if choice == 'y':
                writer.writerow([kalshi_ticker, poly_slug])
                print("Saved.")
            else:
                print("Skipped.")

    print(f"\nDone. Saved matches to '{OUTPUT_FILE}'.")

def main():
    kalshi_markets = get_kalshi_markets()
    poly_markets = get_polymarket_markets()

    if not kalshi_markets or not poly_markets:
        print("\nCould not fetch markets from one or both platforms. Exiting.")
        return
    
    matches = find_similar_markets(kalshi_markets, poly_markets, SIMILARITY_THRESHOLD)
    print(f"\n--- Found {len(matches)} Potential Matches ---")
    
    if not matches:
        print("No strong matches found.")
        return

    matches.sort(key=lambda x: x['score'], reverse=True)
    interactive_save(matches)

if __name__ == "__main__":
    main()
