from transformers import AutoTokenizer

def safety_check(full_prompt, threshold=2000):
    tokenizer = AutoTokenizer.from_pretrained("llmware/dragon-mistral-7b-v0")
    tokenized = tokenizer(full_prompt, return_tensors="pt")
    num_tokens = len(tokenized["input_ids"][0])

    if num_tokens > threshold:
        # Split the prompt to make it less than or equal to 2000 tokens
        tokens = tokenizer.tokenize(full_prompt)
        prompt_parts = []
        current_part = ""

        for token in tokens:
            current_part += token
            if len(tokenizer.encode(current_part, return_tensors="pt")[0]) > threshold:
                prompt_parts.append(current_part)
                current_part = ""

        if current_part:
            prompt_parts.append(current_part)

        return prompt_parts[0]
    else:
        return full_prompt

# full_prompt = "Your long prompt goes here..."
# threshold = 2000
# split_prompts = safety_check(full_prompt, threshold)
# print(split_prompts)
