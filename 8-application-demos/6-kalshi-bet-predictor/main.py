import json
from typing import Tuple
import requests
import re
from dotenv import load_dotenv
import os
from exa_py import Exa
from openai import OpenAI


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


class BetPredictor:
    def __init__(self, model_name: str = "gpt-5-nano"):
        
        load_dotenv()
        
        exa_api_key = os.environ.get("EXA_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not exa_api_key:
            raise EnvironmentError("Missing EXA_API_KEY in environment variables")
        if not openai_api_key: 
            raise EnvironmentError("Missing OPENAI_API_KEY in environment variables")

        self.exa = Exa(exa_api_key)
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

        print(f"Using model: {model_name}")

    def _generate_response(self, prompt: str) -> str:

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )

        output = response.output_text.strip()
        print(f"Generated this response: {output}")

        return output
    
    def convert_rules_to_question(self, rules:str) -> str:
        prompt = (
            "A market resolution statement usually describes the conditions under which a market would resolve to \"Yes\" or \"No\". "
            "Your task is to extract and rewrite the core factual question that determines whether the market will resolve to \"Yes.\"\n"
            "Input format: A sentence describing how a market will resolve.\n"
            "Example: \"Will the market resolve to Yes if Curtis Sliwa drops out of the NYC Mayoral race before Nov 4, 2025?\"\n"
            "Output format: A clear, grammatically correct factual question that reflects the underlying condition of the resolution, removing any meta-language about the market or its resolution.\n"
            "The condition will start with \"if [event], then the market will resolve to \"Yes\" and you must convert it to \"Will [event]?\"\n"
            "Examples:\n"
            "Input: \"If Curtis Sliwa drops out of the NYC Mayoral race before Nov 4, 2025, then the market resolves to Yes\"\n"
            "Output: \"Will Curtis Sliwa drop out of the NYC Mayoral race before Nov 4, 2025?\"\n"
            "Input: \"If Donald Trump wins the 2024 U.S. presidential election, then the market resolves to Yes\"\n"
            "Output: \"Will Donald Trump win the 2024 U.S. presidential election?\"\n"
            "Only output the rewritten factual question with no explanations or commentary.\n"
            f"STATEMENT: {rules}"
        )

        raw_response = self._generate_response(prompt)

        return raw_response

    def get_relevant_questions(self, question: str) -> list[str]:

        prompt = (
            "Based on the following question, generate a list of 5 relevant questions "
            "that one could search online to gather more information. "
            "These questions should yield information that would be helpful to answering "
            "the following question in an objective manner.\n\n"
            "Your response SHOULD ONLY BE the following lines, in this exact format:\n"
            "1. <question 1>\n"
            "2. <question 2>\n"
            "3. <question 3>\n"
            "4. <question 4>\n"
            "5. <question 5>\n"
            "Do not add ANY preamble, conclusion, or extra text.\n\n"
            f"Question: \"{question}\""
        )

        raw_response = self._generate_response(prompt)

        relevant_questions = []
        for line in raw_response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                clean_question = line.split('.', 1)[-1].strip()
                relevant_questions.append(clean_question)
        
        print(f"Generated relevant questions: {relevant_questions}")

        return relevant_questions

    
    def get_information(self, questions):
        results = [self.exa.answer(q, text=True) for q in questions]
        answers = [r.answer for r in results]
        return answers

    def get_binary_answer_with_percentage(self, information: str, question: str) -> Tuple[str, str, str]:
        prompt = (
            "Analyze the provided information below to answer the given binary question. "
            "Based on the information, determine the probability that the answer is 'Yes' or 'No'.\n\n"
            "--- Information ---\n"
            f"{information}\n\n"
            "--- Question ---\n"
            f"{question}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Your response MUST ONLY be a single line in THIS EXACT FORMAT:\n"
            "   Yes: <YES PERCENTAGE>%, No: <NO PERCENTAGE>%, Explanation: <EXPLANATION>\n"
            "2. Percentages must sum to 100%.\n"
            "3. Do NOT include any preamble, summary, or additional text.\n"
            "4. Provide a brief but clear explanation supporting your probabilities.\n\n"
        )

        response = self._generate_response(prompt)

        match = re.search(r"Yes: (.*?), No: (.*?), Explanation: (.*)", response, re.DOTALL)

        if match:
            yes, no, explanation = match.groups()
            return yes.strip(), no.strip(), explanation.strip()
        else:
            raise ValueError(f"Failed to parse LLM response: {response}")
    
    def evaluate(self, question):
        relevant_questions = self.get_relevant_questions(question)
        answers = self.get_information(relevant_questions)

        information = ""
        for i, v in enumerate(relevant_questions):
            information += f"INFORMATION {i+1}: \n"
            information += f"QUESTION {i+1}: {v}\n"
            information += f"ANSWER {i+1}: {answers[i]} \n\n"

        information.rstrip("\n")
        
        yes, no, explanation = self.get_binary_answer_with_percentage(information, question)
        return yes, no, explanation


predictor = BetPredictor()

def predict(kalshi_ticker, poly_slug):
    kalshi_market = getMarket(True, kalshi_ticker)
    poly_market = getMarket(False, poly_slug)
    question = poly_market['question'] # we use polymarket because they have direct question

    kalshi_real_yes = float(kalshi_market['market']['yes_ask'])
    kalshi_real_no = float(kalshi_market['market']['no_ask'])
    
    poly_values = json.loads(poly_market['outcomePrices'])
    poly_real_yes, poly_real_no = [float(v) for v in poly_values]

    print(f"Question: {question}") 
    
    pred_yes, pred_no, explanation = predictor.evaluate(question)

    match_yes = re.search(r"(\d+)%", pred_yes)
    match_no = re.search(r"(\d+)%", pred_no)
    pred_yes = float(match_yes.group(1))
    pred_no = float(match_no.group(1))

    kalshi_real_yes = float(kalshi_market['market']['yes_ask'])
    kalshi_real_no = float(kalshi_market['market']['no_ask'])
    
    poly_values = json.loads(poly_market['outcomePrices'])
    poly_real_yes, poly_real_no = [float(v)*100 for v in poly_values]

    kalshi_buy_yes = kalshi_real_yes < pred_yes
    kalshi_buy_no = kalshi_real_no < pred_no

    poly_buy_yes = poly_real_yes < pred_yes
    poly_buy_no = poly_real_no < pred_no

    return {
        "kalshi": {
            "buy_yes":kalshi_buy_yes,
            "buy_no": kalshi_buy_no,
            "edge": max(pred_yes-kalshi_real_yes, pred_no-kalshi_real_no),   
        },
        "polymarket": {
            "buy_yes":poly_buy_yes,
            "buy_no": poly_buy_no,
            "edge": max(pred_yes-poly_real_yes, pred_no-poly_real_no),
        },
        "yes": pred_yes,
        "no": pred_no,
        "explanation": explanation
    }