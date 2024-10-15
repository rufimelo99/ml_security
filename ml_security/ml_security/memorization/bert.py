"""
    Follows the idea in https://arxiv.org/pdf/2012.07805
"""

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml_security.datasets.nlp.nlp import parse_commoncrawl
from ml_security.logger import logger
from ml_security.utils.nlp_utils import calculate_perplexity
from ml_security.utils.utils import get_device

DEVICE = get_device(allow_mps=False)


logger.info("Loading GPT-2 model")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memorization attack")
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Sample from top k tokens"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for generation"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="memorization/output.json",
        help="Output file",
    )

    args = parser.parse_args()

    cc = parse_commoncrawl()

    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    with tqdm(total=args.num_samples) as pbar:
        for i in range(num_batches):

            input_len = 10
            input_ids = []
            attention_mask = []
            while len(input_ids) < args.batch_size:

                r = np.random.randint(0, len(cc))
                prompt = " ".join(cc[r : r + 100].split(" ")[1:-1])
                inputs = tokenizer(
                    prompt, return_tensors="pt", max_length=input_len, truncation=True
                )
                if len(inputs["input_ids"][0]) == input_len:
                    input_ids.append(inputs["input_ids"][0])
                    attention_mask.append(inputs["attention_mask"][0])

                inputs = {
                    "input_ids": torch.stack(input_ids),
                    "attention_mask": torch.stack(attention_mask),
                }

            prompts = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            output_sequences = model.generate(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=inputs["attention_mask"].to(DEVICE),
                max_length=input_len + args.seq_len,
                do_sample=True,
                top_k=args.top_k,
                top_p=1.0,
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                p1 = calculate_perplexity(text, model, tokenizer, DEVICE)
                p_lower = calculate_perplexity(text.lower(), model, tokenizer, DEVICE)

                # # Zlib "entropy" of sample
                # import zlib
                # zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                logger.info("Perplexity", value=p1.item())
                logger.info("Perplexity lower", value=p_lower.item())
                logger.info("Prompt", value=prompt[:100])
                logger.info("Text", value=text[:100])

                with open(args.output_file, "a") as f:
                    json.dump(
                        {
                            "perplexity": p1.item(),
                            "perplexity_lower": p_lower.item(),
                            "prompt": prompt,
                            "text": text,
                        },
                        f,
                    )
                    f.write("\n")
