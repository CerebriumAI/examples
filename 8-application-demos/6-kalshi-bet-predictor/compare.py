import csv
import json
from typing import Dict, List, Tuple
import asyncio
import aiohttp

def load_markets(csv_path: str) -> List[Tuple[str, str]]:
    markets = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) >= 2:
                markets.append((row[0], row[1]))
    return markets

async def get_market_data(session: aiohttp.ClientSession, kalshi_ticker: str, 
                         poly_slug: str, endpoint_url: str) -> Dict:
    
    payload = json.dumps({
        'kalshi_ticker': kalshi_ticker,
        'poly_slug': poly_slug
    })
    
    headers = {
        'Authorization': '<YOUR AUTHORIZATION>',
        'Content-Type': 'application/json'
    }
    
    try:
        async with session.post(endpoint_url, headers=headers, data=payload) as response:
            response.raise_for_status()
            data = await response.json()
            print(data)
            data = data['result']
            
            kalshi_data = data['kalshi']
            poly_data = data['polymarket']
            
            return {
                'kalshi_ticker': kalshi_ticker,
                'poly_slug': poly_slug,
                'kalshi_edge_value': kalshi_data['edge'],
                'poly_edge_value': poly_data['edge'],
                'kalshi_is_buy_yes': kalshi_data['buy_yes'],
                'kalshi_is_buy_no': kalshi_data['buy_no'],
                'poly_is_buy_yes': poly_data['buy_yes'],
                'poly_is_buy_no': poly_data['buy_no'],
            }
    except Exception as e:
        print(f"Error fetching data for {kalshi_ticker}/{poly_slug}: {e}")
        return None

async def analyze_markets_async(csv_path: str, endpoint_url: str) -> List[Dict]:
    markets = load_markets(csv_path)
    
    print(f"Fetching data for {len(markets)} markets all at once...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [get_market_data(session, kalshi_ticker, poly_slug, endpoint_url) 
                for kalshi_ticker, poly_slug in markets]
        
        results = await asyncio.gather(*tasks)
    
    return [r for r in results if r is not None]

def compute_statistics(results: List[Dict]) -> None:
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    if not results:
        print("No results to analyze")
        return
    
    total_markets = len(results)
    
    kalshi_edges_values = [r['kalshi_edge_value'] for r in results]
    kalshi_edge_sum = sum(kalshi_edges_values)
    
    poly_edges_values = [r['poly_edge_value'] for r in results]
    poly_edge_sum = sum(poly_edges_values)
    
    kalshi_better_count = sum(1 for r in results if r['kalshi_edge_value'] > r['poly_edge_value'])
    poly_better_count = sum(1 for r in results if r['poly_edge_value'] > r['kalshi_edge_value'])
    equal_count = total_markets - kalshi_better_count - poly_better_count
    
    edge_differences = [abs(r['kalshi_edge_value'] - r['poly_edge_value']) for r in results]
    avg_edge_difference = sum(edge_differences) / total_markets
    max_edge_difference = max(edge_differences)
    
    print(f"\nTotal markets analyzed: {total_markets}")
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    print(f"Markets with greater Kalshi edge:      {kalshi_better_count} ({kalshi_better_count/total_markets*100:.1f}%)")
    print(f"Markets with greater Polymarket edge:  {poly_better_count} ({poly_better_count/total_markets*100:.1f}%)")
    print(f"Markets with equal edge:               {equal_count} ({equal_count/total_markets*100:.1f}%)")
    print(f"\nAverage edge difference: {avg_edge_difference:.4f} cents")
    print(f"Max edge difference:     {max_edge_difference:.4f} cents")
    
    print("\n" + "="*80)
    if kalshi_edge_sum > poly_edge_sum:
        advantage = kalshi_edge_sum - poly_edge_sum
        print(f"OVERALL: Kalshi has greater total edge (+{advantage:.4f}) cents")
        print(f"OVERALL: Kalshi has an average edge of (+{advantage/total_markets:.4f}) cents per market")
    elif poly_edge_sum > kalshi_edge_sum:
        advantage = poly_edge_sum - kalshi_edge_sum
        print(f"OVERALL: Polymarket has greater total edge (+{advantage:.4f}) cents")
        print(f"OVERALL: Polymarket has an average edge of (+{advantage/total_markets:.4f}) cents per market")
    else:
        print(f"OVERALL: Both platforms have equal total edge")
    print("="*80)

def main():
    CSV_PATH = "<PATH_TO_YOUR_CSV_FILE>"
    ENDPOINT_URL = '<YOUR_CEREBRIUM_PREDICT_URL>'
    
    print("Starting async market analysis...")
    results = asyncio.run(analyze_markets_async(CSV_PATH, ENDPOINT_URL))
    
    print(f"\nSuccessfully fetched {len(results)} markets")
    
    compute_statistics(results)

if __name__ == "__main__":    
    main()