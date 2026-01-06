import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import InvalidOperation
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import string
import secrets

char_pool = string.ascii_letters + string.digits + string.punctuation

def getRandomString(length) -> str:
    return ''.join(secrets.choice(char_pool) for _ in range(length))

def positional_mirror_cipher(text):
    alphabet = string.ascii_lowercase
    mirror = alphabet[::-1]
    result = []
    
    for i, char in enumerate(text):
        if char.lower() in alphabet:
            is_upper = char.isupper()
            idx = alphabet.index(char.lower())
            
            if i % 2 == 0: # Even: Mirror
                new_char = mirror[idx]
            else:          # Odd: Shift +3
                new_char = alphabet[(idx + 3) % 26]
                
            result.append(new_char.upper() if is_upper else new_char)
        else:
            result.append(char)
            
    return "".join(result)

def generateCipher(length: int) -> Tuple[str, str]:
    plain_text = getRandomString(length)
    cipher_text = positional_mirror_cipher(plain_text)
    return plain_text, cipher_text


class LLMResponseError(Exception):
    pass

class LLMSolver:
    def __init__(self, model_name: str = "gpt-5-nano"):
        load_dotenv()

        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY in environment variables")

        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

        print(f"Using model: {model_name}")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )

            output_text = response.output_text.strip()
            print(f"Generated raw response: {output_text}")

            return output_text

        except (InvalidOperation, AttributeError, KeyError) as parse_error:
            raise LLMResponseError(
                "LLM response could not be interpreted as a number."
            ) from parse_error
        except Exception as api_error:
            raise LLMResponseError(f"Error during API call: {api_error}") from api_error


def _write_result_rows(
    writer: csv.writer,
    length: int,
    results: List[Tuple[int, str, str, str, bool]],
) -> None:
    for _, expression, truthAns, llmAns, is_correct in results:
        writer.writerow(
            [
                length,
                expression,
                truthAns,
                llmAns,
                is_correct,
            ]
        )

def evaluate(
    length: int,
    iterations: int = 100,
    csv_path: str = "llm_results.csv",
    writer: Optional[csv.writer] = None,
) -> Tuple[int, int, int]:
    llm = LLMSolver()

    csv_full_path: Optional[Path] = None
    if writer is None:
        csv_full_path = Path(csv_path).expanduser().resolve()
        csv_full_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Tuple[int, str, str]] = []
    for i in range(1, iterations+1):
        plain, cipher = generateCipher(length)
        samples.append((i, plain, cipher))

    samples_lookup = {
        run_index: (expression, truthAns)
        for run_index, expression, truthAns in samples
    }

    def _solve_sample(
        run_index: int, expression: str, truthAns: str 
    ) -> Tuple[int, str, str, str, bool]:
        prompt = (
            "Consider the following cipher. The Rules. Consider (0) indexing the entire string.\n"
            "Even Index ($0, 2, 4...$): Replace with the 'mirror' of the alphabet ($a \\to z, A \\to Z, b \\to y, B \\to Y, c \\to x$, etc.).\n"
            "Odd Index ($1, 3, 5...$): Shift forward by 3 (a \\to d, A \\to D, b \\to e, B \\to E, z \\to c$).\n"
            "All other characters: Leave unchanged.\n"
            f"Please cipher the following:\n{expression} \nOutput ONLY the final ciphered text with no additional commentary or punctuation."
        )

        llmAns = llm.generate_response(prompt)
        is_correct = llmAns == truthAns
        print(f"Found LLM answer: {llmAns} (Correct: {is_correct})")
        return run_index, expression, truthAns, llmAns, is_correct

    results: List[Tuple[int, str, str, str, bool]] = []
    max_workers = min(10, iterations)
    processed = 0
    skipped_errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_solve_sample, run_index, expression, truthAns): run_index
            for run_index, expression, truthAns in samples
        }

        for future in as_completed(future_map):
            run_index = future_map[future]
            try:
                (
                    _,
                    expression,
                    truthAns,
                    llmAns,
                    is_correct,
                ) = future.result()
            except LLMResponseError as error:
                expression, truthAns = samples_lookup[run_index]
                snippet = (
                    f"{expression[:60]}..."
                    if len(expression) > 60
                    else expression
                )
                processed += 1
                skipped_errors += 1
                print(
                    f"[{processed}/{iterations}] LLM error for run #{run_index}: {error}. "
                    f"Skipping expression: {snippet}"
                )
                continue
            except Exception as error:
                expression, truthAns = samples_lookup[run_index]
                snippet = (
                    f"{expression[:60]}..."
                    if len(expression) > 60
                    else expression
                )
                processed += 1
                skipped_errors += 1
                print(
                    f"[{processed}/{iterations}] Unexpected error for run #{run_index}: {error}. "
                    f"Skipping expression: {snippet}"
                )
                continue

            results.append((run_index, expression, truthAns, llmAns, is_correct))
            processed += 1

            print(
                f"[{processed}/{iterations}] "
                f"Truth={truthAns} "
                f"LLM={llmAns} "
                f"({'Correct' if is_correct else 'Incorrect'})"
            )

    results.sort(key=lambda item: item[0])
    correct_count = sum(int(item[4]) for item in results)
    attempts = len(results)

    if writer is None:
        assert csv_full_path is not None
        with csv_full_path.open("w", newline="") as csvfile:
            file_writer = csv.writer(csvfile)
            file_writer.writerow(
                ["length", "original_text", "true_cipher", "model_cipher", "is_correct"]
            )
            _write_result_rows(file_writer, length, results)
    else:
        _write_result_rows(writer, length, results)

    if attempts:
        accuracy = (correct_count / attempts) * 100
        print(
            f"LLM accuracy for length {length}: {accuracy:.2f}% "
            f"({correct_count}/{attempts}) with {skipped_errors} skipped."
        )
    else:
        print(
            f"LLM produced no successful runs for length {length}. "
            f"Skipped {skipped_errors} attempts."
        )

    return correct_count, attempts, skipped_errors


def run_length_sweep(
    mn: int = 4,
    mx: int = 10,
    iterations_per_length: int = 2,
    csv_path: str = "llm_results.csv",
) -> None:
    lengths = [i for i in range(mn, mx + 1)]
    csv_full_path = Path(csv_path).expanduser().resolve()
    csv_full_path.parent.mkdir(parents=True, exist_ok=True)

    overall_correct = 0
    overall_attempts = 0
    overall_skipped = 0

    with csv_full_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["length", "original_text", "true_cipher", "model_cipher", "is_correct"]
        )

        for length in lengths:
            print(f"\n=== Evaluating expressions with length {length} ===")
            correct, attempts, skipped = evaluate(
                length=length,
                iterations=iterations_per_length,
                writer=writer,
            )
            overall_correct += correct
            overall_attempts += attempts
            overall_skipped += skipped

    if overall_attempts:
        overall_accuracy = (overall_correct / overall_attempts) * 100
        print(
            f"\nOverall accuracy across lengths {lengths[0]}-{lengths[-1]}: "
            f"{overall_accuracy:.2f}% "
            f"({overall_correct}/{overall_attempts}) with {overall_skipped} skipped."
        )
    else:
        print(
            f"\nNo successful evaluations recorded across lengths {lengths[0]}-{lengths[-1]}. "
            f"Skipped {overall_skipped} attempts."
        )


if __name__ == "__main__":
    run_length_sweep(1, 3, 3, "llm_results.csv") # from length 1 to 6, 200 iterations each