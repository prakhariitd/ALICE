import os
import openai
from transformers import GPT2TokenizerFast
import numpy as np

def classify(tweet):
    openai.api_key = "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"

    # Load the tokenizer.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Make sure the labels are formatted correctly.
    labels = ["True", "False", "Unclassified"]
    labels = [label.strip().lower().capitalize() for label in labels]

    # Encode the labels with extra white space prepended.
    labels_tokens = {label: tokenizer.encode(" " + label) for label in labels}
    print(labels_tokens)


    # file_res = openai.File.create(file=open("openai-quickstart-python/train.jsonl"), purpose="classifications")

    # Take the starting tokens for probability estimation.
    # Labels should have distinct starting tokens.
    # Here tokens are case-sensitive.
    first_token_to_label = {tokens[0]: label for label, tokens in labels_tokens.items()}
    logit_bias = {str(first_token): 100 for first_token in first_token_to_label}

    result = openai.Classification.create(
        file= "file-O3NWJpJsob3jzCaKicg4XLcj", #file with 3 labels
        query=tweet,
        search_model="ada", 
        model="curie", 
        max_examples=200,
        labels=labels,
        logprobs=4,  # Here we set it to be len(labels) + 1, but it can be larger.
        expand=["completion"],
        logit_bias = logit_bias
    )

    print (result)

    top_logprobs = result["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
    token_probs = {
        tokenizer.encode(token)[0]: np.exp(logp) 
        for token, logp in top_logprobs.items()
    }
    label_probs = {
        first_token_to_label[token]: prob 
        for token, prob in token_probs.items()
        if token in first_token_to_label
    }

    # Fill in the probability for the special "Unknown" label.
    if sum(label_probs.values()) < 1.0:
        label_probs["Unknown"] = 1.0 - sum(label_probs.values())

    print(label_probs)
    label_probs_vals = [0]*3
    label_probs_vals[0] = label_probs["True"]
    label_probs_vals[1] = label_probs["False"]
    label_probs_vals[2] = label_probs["Unclassified"]
    return label_probs_vals

if __name__ == "__main__":
    tweet = "israel economy is bad"
    classify(tweet)