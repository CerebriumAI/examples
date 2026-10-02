# Kalshi Bet Predictor
This repository contains a set of Python scripts designed to find equivalent binary markets across Kalshi and Polymarket, use an LLM via OpenAI and web search via Exa to generate an independent prediction, and then calculate the trading "edge" on both platforms.

Core Components
---------------

The project is structured around three main scripts:

**`find_equiv_markets.py`**: A utility script to automatically search Kalshi and Polymarket APIs, use a Sentence Transformer and FAISS vector index to find markets with similar titles (i.e., equivalent questions), and save potential matches to a CSV file for manual review.
    
**`analyst.py`** and **`main.py`**: These scripts form the core prediction engine.
    
*   `analyst.py` handles API interactions with OpenAI (for prediction and question generation) and Exa (for web information retrieval).
    
*   `main.py` fetches the current prices from both Kalshi and Polymarket, runs the prediction via `BetAnalyst`, and calculates the trading edge against the model's prediction. This logic is intended to be hosted on Cerebrium
        
*   **`compare.py`**: This script reads the market pairs from the CSV, asynchronously calls the hosted prediction endpoint for each pair, and compiles statistics on the trading edge and which platform offers a better opportunity more frequently.
    


Prerequisites
-------------

You will need API keys for the following services:

*   **OpenAI**: For the large language model (`BetAnalyst` class).
    
*   **Exa**: For semantic search/information retrieval (`BetAnalyst` class).
    
*   **Cerebrium** (or similar hosting platform): To deploy the `main.py` and `analyst.py` logic as a prediction endpoint.
    

Create a `.env` file in your project root to store your keys:

``` OPENAI_API_KEY="your_openai_key" EXA_API_KEY="your_exa_key" ```

Setup and Installation
----------------------

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```


Workflow
--------

1. Host the prediction service by deploying `main.py` and `analyst.py` on Cerebrium to expose a `predict` endpoint that runs the `BetAnalyst` logic.
2. Run `find_equiv_markets.py` to identify equivalent Kalshi and Polymarket markets and export the candidate pairs to a CSV file.
3. Execute `compare.py`, which loads the CSV pairs, calls the hosted prediction endpoint for each pair, and aggregates the edge statistics to highlight the most favorable markets.

