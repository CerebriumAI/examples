import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import requests
import re
from dotenv import load_dotenv
import os
from exa_py import Exa

def getKalshiQuestion(market_ticker)->Tuple[str,str]:
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        obj = res.json()
        return obj['market']['rules_primary']
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching Kalshi market data: {e}")

def getKalshiOdds(market_ticker)->Tuple[str, str]:
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        obj = res.json()
        return obj['market']['yes_ask'], obj['market']['no_ask']
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching Kalshi market data: {e}")


class BetPredictor:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.bfloat16,
            device_map="auto"
        )

        load_dotenv()

        self.exa = Exa(os.environ.get("EXA_API_KEY"))

        print(f"Loaded model {model_name}!")

    def _generate_response(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids_len = inputs['input_ids'].shape[-1]

        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        newly_generated_ids = output_sequences[0, input_ids_len:]
        
        response = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True).strip()
        
        print(f"Generated this response! {response}")
        return response
    
    def convert_rules_to_question(self, rules:str) -> str:
        prompt = (
            "You will receive a sentence that is a statement of the following type:"
            "If <conditional>, then the market resolves to Yes"
            "Convert the conditional to a yes/no question"
            "Your response SHOULD ONLY BE a SINGLE line consisting of the yes/no question:\n"
            "Do not add ANY preamble, conclusion, or extra text.\n\n"
            f"STATEMENT: {rules}\n"
        )

        raw_response = self._generate_response(prompt, max_new_tokens=400)

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
            f"Question: \"{question}\"\n"
        )

        raw_response = self._generate_response(prompt, max_new_tokens=400)

        relevant_questions = []
        for line in raw_response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                clean_question = line.split('.', 1)[-1].strip()
                relevant_questions.append(clean_question)
        
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
            "AGAIN, Your response MUST ONLY be a single line in THIS EXACT FORMAT: Yes: <YES PERCENTAGE>%, No: <NO PERCENTAGE>%, Explanation: <EXPLANATION>"
        )

        response = self._generate_response(prompt, max_new_tokens=800)

        match = re.search(r"Yes: (.*?), No: (.*?), Explanation: (.*)", response, re.DOTALL)

        if match:
            yes, no, explanation = match.groups()
            return yes.strip(), no.strip(), explanation.strip()
        else:
            raise ValueError(f"Failed to parse LLM response: {response}")
    
    def predict(self, question):
        relevant_questions = self.get_relevant_questions(question)
        answers = self.get_information(relevant_questions)

        information = ""
        for i, v in enumerate(relevant_questions):
            information += f"INFORMATION {i+1}: \n"
            information += f"QUESTION {i+1}: {v}\n"
            information += f"ANSWER {i+1}: {answers[i]} \n\n"
        
        yes, no, explanation = self.get_binary_answer_with_percentage(information, question)
        return yes, no, explanation


predictor = BetPredictor()

def predict(ticker: str):
    rules = getKalshiQuestion(ticker)
    question = predictor.convert_rules_to_question(rules)
    
    predYes, predNo, explanation = predictor.predict(question)

    realYes, realNo = getKalshiOdds(ticker)
    
    if realYes < predYes: # undervalued
        buyYes = True
    if realNo < predNo: # undervalued
        buyNo = True

    return {"buy_yes":buyYes, "buy_no": buyNo, "yes": predYes, "no": predNo, "explanation": explanation}

