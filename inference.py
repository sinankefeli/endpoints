#!/usr/bin/env python3
"""
This script performs inference using the vLLM (very Large Language Model) 
for job description generation.


Data:
Input use clean job descriptions from 1024 job postings
Input average tokens: 724.2 / description

vLLM:
Avg generation throughput: 1000 token/s - 1700 tokens/s
time elapsed: 236.67 seconds
"""
import pandas as pd
from openai import OpenAI
from time import time
from transformers import AutoTokenizer

PROMPT = """
<s>
You are a helpful assistant.
</s>
[INST]
Provide a brief summary of the job description:
{context}
"""

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

if __name__ == "__main__":
    # read data
    data = pd.read_csv("data/clean_desc.csv", dtype=str)
    # create prompt
    prompt = [PROMPT.format(context=row["clean_desc"]) for _, row in data.iterrows()]
    # count number of tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_tokens = [len(tokenizer.encode(p)) for p in prompt]
    # connect to client
    client_openai = OpenAI(api_key="EMPTY", base_url="http://localhost:8002/v1")
    # llm
    ti = time()
    output = client_openai.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=2048,
        temperature=0.1,
    )
    tf = time()
    print(f"time elapsed: {tf-ti:2.2f} seconds")
    # output
    results = [_o["text"] for _o in output.dict()["choices"]]
    