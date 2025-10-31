
from typing import Tuple
from dotenv import load_dotenv
import os
import json
from exa_py import Exa
from openai import OpenAI
from pydantic import BaseModel

class BetAnalyst:
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

    def _generate_response(self, prompt: str, text_format = None):
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )

            output_text = response.output_text.strip()
            print(f"Generated raw response: {output_text}")

            if text_format is not None:
                parsed = self.client.responses.parse(
                    model=self.model_name,
                    input=[
                        {
                            "role": "user",
                            "content": output_text
                        },
                    ],
                    text_format=text_format,
                )
                print(f"Parsed structured response: {parsed.output_parsed}")
                return parsed.output_parsed

            return output_text

        except Exception as e:
            raise RuntimeError(f"Error during API call: {e}") from e
    
    def convert_market_to_resolution(self, rules:str) -> str:
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

    
    def get_web_info(self, questions):
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
        
        class Response(BaseModel):
            yes_percentage: str
            no_percentage: str
            explanation: str

        response = self._generate_response(prompt, Response)
        print(f"HELLO {response}")

        try:
            return response.yes_percentage, response.no_percentage, response.explanation
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse output as JSON: {response}")
    