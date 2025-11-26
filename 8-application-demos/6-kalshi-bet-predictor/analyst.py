from typing import Tuple
from dotenv import load_dotenv
import os
import json
from exa_py import Exa
from openai import OpenAI
from pydantic import BaseModel

class BetAnalyst:
    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initializes the API clients and loads necessary API keys from environment variables"""
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
        """Sends a prompt to the OpenAI API and optionally parses the output into a structured format"""
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )

            output_text = response.output_text.strip()
            print(f"Generated raw response: {output_text}")

            if text_format is not None:
                # If a Pydantic model (text_format) is provided, re-parse the raw output into that structure.
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

    def get_relevant_questions(self, question: str) -> list[str]:
        """Generates a list of related search queries based on an initial user question"""
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
                # Parse lines like "1. What is..." into "What is..."
                clean_question = line.split('.', 1)[-1].strip()
                relevant_questions.append(clean_question)
        
        print(f"Generated relevant questions: {relevant_questions}")

        return relevant_questions

    
    def get_web_info(self, questions):
        """Uses the Exa API to find answers for a list of questions."""
        results = [self.exa.answer(q, text=True) for q in questions]
        answers = [r.answer for r in results]
        return answers

    def get_binary_answer_with_percentage(self, information: str, question: str) -> Tuple[str, str, str]:
        """Analyzes provided information to return a Yes/No probability and explanation for a given question"""
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
        
        # Define the expected Pydantic structure for the _generate_response 'text_format' parameter
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
    