import csv
import requests
from typing import Dict, List, Tuple
import asyncio
import aiohttp

def load_markets(csv_path: str) -> List[Tuple[str, str]]:
    markets = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if present
        for row in reader:
            if len(row) >= 2:
                markets.append((row[0], row[1]))
    return markets

async def get_market_data(session: aiohttp.ClientSession, kalshi_id: str, 
                         polymarket_slug: str, endpoint_url: str) -> Dict:
    
    payload = {
        'kalshi_id': kalshi_id,
        'polymarket_slug': polymarket_slug
    }
    
    try:
        async with session.post(endpoint_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            data = await response.json()
            
            kalshi_data = data['Kalshi']
            polymarket_data = data['Polymarket']
            
            return {
                'kalshi_id': kalshi_id,
                'polymarket_slug': polymarket_slug,
                'kalshi_edge': kalshi_data['edge'],
                'polymarket_edge': polymarket_data['edge'],
                'kalshi_buy_yes': kalshi_data['buy_yes'],
                'kalshi_buy_no': kalshi_data['buy_no'],
                'polymarket_buy_yes': polymarket_data['buy_yes'],
                'polymarket_buy_no': polymarket_data['buy_no'],
            }
    except Exception as e:
        print(f"Error fetching data for {kalshi_id}/{polymarket_slug}: {e}")
        return None

async def analyze_markets_async(csv_path: str, endpoint_url: str) -> List[Dict]:
    markets = load_markets(csv_path)
    
    print(f"Fetching data for {len(markets)} markets all at once...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [get_market_data(session, kalshi_id, polymarket_slug, endpoint_url) 
                for kalshi_id, polymarket_slug in markets]
        
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
    
    kalshi_edges = [r['kalshi_edge'] for r in results]
    total_kalshi_edge = sum(kalshi_edges)
    
    polymarket_edges = [r['polymarket_edge'] for r in results]
    total_polymarket_edge = sum(polymarket_edges)
    
    kalshi_better_count = sum(1 for r in results if r['kalshi_edge'] > r['polymarket_edge'])
    polymarket_better_count = sum(1 for r in results if r['polymarket_edge'] > r['kalshi_edge'])
    equal_count = total_markets - kalshi_better_count - polymarket_better_count
    
    edge_differences = [abs(r['kalshi_edge'] - r['polymarket_edge']) for r in results]
    avg_edge_difference = sum(edge_differences) / total_markets
    max_edge_difference = max(edge_differences)
    
    # Results
    print(f"\nTotal markets analyzed: {total_markets}")
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    print(f"Markets with greater Kalshi edge:      {kalshi_better_count} ({kalshi_better_count/total_markets*100:.1f}%)")
    print(f"Markets with greater Polymarket edge:  {polymarket_better_count} ({polymarket_better_count/total_markets*100:.1f}%)")
    print(f"Markets with equal edge:               {equal_count} ({equal_count/total_markets*100:.1f}%)")
    print(f"\nAverage edge difference: {avg_edge_difference:.4f}")
    print(f"Max edge difference:     {max_edge_difference:.4f}")
    
    # Overall winner
    print("\n" + "="*80)
    if total_kalshi_edge > total_polymarket_edge:
        advantage = total_kalshi_edge - total_polymarket_edge
        print(f"OVERALL: Kalshi has greater total edge (+{advantage:.4f})")
    elif total_polymarket_edge > total_kalshi_edge:
        advantage = total_polymarket_edge - total_kalshi_edge
        print(f"OVERALL: Polymarket has greater total edge (+{advantage:.4f})")
    else:
        print(f"OVERALL: Both platforms have equal total edge")
    print("="*80)

def main():
    CSV_PATH = 'markets.csv' 
    ENDPOINT_URL = 'https://{cerebrium}/predict' # Your hosted endpoint
    
    print("Starting async market analysis...")
    results = asyncio.run(analyze_markets_async(CSV_PATH, ENDPOINT_URL))
    
    print(f"\nSuccessfully fetched {len(results)} markets")
    
    compute_statistics(results)

if __name__ == "__main__":
    main()