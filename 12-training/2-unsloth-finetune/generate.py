import string
import secrets
import json
import random

def positional_mirror_cipher_with_logic(text):
    alphabet = string.ascii_lowercase
    mirror = alphabet[::-1]
    
    analysis_steps = []
    final_result = []
    
    for i, char in enumerate(text):
        if char.lower() in alphabet:
            is_upper = char.isupper()
            idx = alphabet.index(char.lower())
            
            if i % 2 == 0:
                # Even: Mirror (a -> z)
                new_char = mirror[idx]
                transformation = f"{char}({i}:even)->{new_char.upper() if is_upper else new_char}"
            else:
                # Odd: Shift +3 (a -> d)
                new_char = alphabet[(idx + 3) % 26]
                transformation = f"{char}({i}:odd)->{new_char.upper() if is_upper else new_char}"
                
            res_char = new_char.upper() if is_upper else new_char
            final_result.append(res_char)
            analysis_steps.append(transformation)
        else:
            # Other (Spaces, Digits, Punctuation)
            final_result.append(char)
            analysis_steps.append(f"{char}({i}:sym)->{char}")
            
    analysis_str = ", ".join(analysis_steps)
    cipher_text = "".join(final_result)
    return analysis_str, cipher_text

def getRandomString(length) -> str:
    categories = [
        string.ascii_lowercase,
        string.ascii_uppercase,
        string.digits,
        string.punctuation + " " 
    ]
    # distribution for each of the above categories
    weights = [70, 15, 5, 10]
    
    result = []
    for _ in range(length):
        chosen_category = random.choices(categories, weights=weights, k=1)[0]
        result.append(secrets.choice(chosen_category))
        
    return "".join(result)

def create_trainset(filename="./results/dataset.jsonl", num_samples=20000):
    instruction = (
        "Apply the positional mirror-shift cipher to the input text using a GLOBAL index. "
        "The index starts at 0 for the first character and increments for EVERY character (including spaces and symbols). "
        "Rules: If the global index is even, mirror the letter ($a \\to z, A \\to Z, b \\to y, B \\to Y, c \\to x$, etc.). "
        "If the global index is odd, shift the letter forward by 3 (a \\to d, A \\to D, b \\to e, B \\to E, z \\to c$). "
        "Non-alphabetic characters do not change but still consume an index count."
    )

    with open(filename, "w") as f:
        for _ in range(num_samples):
            length = secrets.randbelow(5) + 5  
            plain_text = getRandomString(length)
            analysis, cipher = positional_mirror_cipher_with_logic(plain_text)
            
            response = f"Analysis: {analysis}\nFinal Cipher: {cipher}"
            
            record = {
                "text": f"### Instruction:\n{instruction}\n\n### Input:\n{plain_text}\n\n### Response:\n{response}"
            }
            f.write(json.dumps(record) + "\n")

    print(f"Successfully generated {num_samples} samples with the requested character distribution.")

def create_testset(filename="./results/testset.json", num_samples=200):
    with open(filename, "w") as f:
        plain_texts = []
        ciphered_texts = []
        for _ in range(num_samples):
            length = secrets.randbelow(5) + 5 # length between 5 and 9
            plain_text = getRandomString(length)
            _, cipher = positional_mirror_cipher_with_logic(plain_text)
            plain_texts.append(plain_text)
            ciphered_texts.append(cipher)
        
        json.dump({"plain_texts": plain_texts,
                   "ciphered_texts": ciphered_texts}, f)
    print(f"Successfully generated {num_samples} test samples.")

if __name__ == "__main__":
    create_trainset()
    # create_testset()