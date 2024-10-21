"""
    Inspired by https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb.


Check https://arxiv.org/abs/1902.04094 for more details. 
@article{wang2019bert,
  title={BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model},
  author={Wang, Alex and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1902.04094},
  year={2019}
}
"""

import argparse
import math

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ml_security.utils.utils import get_device


def tokenize_batch(batch):
    """Tokenizes a batch of text"""
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    """Untokenizes a batch of text"""
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def detokenize(sent):
    """Roughly detokenizes (mainly undoes wordpiece)"""
    new_sent = []
    for _, tok in enumerate(sent):
        if not tok:
            continue
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


def generate_step(
    out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True
):
    """Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out.logits[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1):
    """Get initial sentence by padding seed_text with either masks or random words to max_len"""
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch)


def parallel_sequential_generation(
    seed_text,
    mask_id: int,
    max_len=15,
    top_k=0,
    temperature=None,
    max_iter=300,
    burnin=200,
    cuda=False,
    print_every=10,
    verbose=True,
    batch_size=2,
):
    """Generates for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(
            out,
            gen_idx=seed_len + kk,
            top_k=topk,
            temperature=temperature,
            sample=(ii < burnin),
        )
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = idxs[jj]

        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = (
                for_print[: seed_len + kk + 1]
                + ["(*)"]
                + for_print[seed_len + kk + 1 :]
            )

    return untokenize_batch(batch)


def sequential_generation(
    seed_text,
    sep_id: int,
    batch_size=2,
    max_len=15,
    leed_out_len=15,
    top_k=0,
    temperature=None,
    sample=True,
    cuda=False,
):
    """Generate one word at a time, in L->R order"""
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    batch = batch.cuda() if cuda else batch

    for ii in range(max_len):
        inp = [sent[: seed_len + ii + leed_out_len] + [sep_id] for sent in batch]
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        idxs = generate_step(
            out,
            gen_idx=seed_len + ii,
            top_k=top_k,
            temperature=temperature,
            sample=sample,
        )
        for jj in range(batch_size):
            batch[jj][seed_len + ii] = idxs[jj]

        return untokenize_batch(batch)


def generate(
    n_samples,
    mask_id: int,
    sep_id: int,
    seed_text="[CLS]",
    batch_size=10,
    max_len=25,
    sample=True,
    top_k=100,
    temperature=1.0,
    burnin=200,
    max_iter=500,
    cuda=False,
    print_every=1,
):
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    for _ in range(n_batches):
        batch = parallel_sequential_generation(
            seed_text,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False,
            mask_id=mask_id,
        )

        # batch = sequential_generation(seed_text, batch_size=20, max_len=max_len, top_k=top_k, temperature=temperature, leed_out_len=leed_out_len, sample=sample)
        sentences += batch
    return sentences


# Utility functions
def write_sents(out_file, sents, should_detokenize=False):
    with open(out_file, "w") as out_fh:
        for sent in sents:
            sent = detokenize(sent[1:-1]) if should_detokenize else sent
            out_fh.write("%s\n" % " ".join(sent))


def score(sentence, model, tokenizer):
    # https://arxiv.org/abs/1910.14659
    tensor_input = tokenizer(sentence, return_tensors="pt", truncation=True)
    repeat_input = tensor_input["input_ids"].repeat(
        tensor_input["input_ids"].size(-1) - 2, 1
    )
    mask = torch.ones(tensor_input["input_ids"].size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default="bert-base-uncased")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--leed_out_len", type=int, default=5)
    parser.add_argument("--burnin", type=int, default=250)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()

    model_version = args.model_version
    n_samples = args.n_samples
    batch_size = args.batch_size
    max_len = args.max_len
    top_k = args.top_k
    temperature = args.temperature
    leed_out_len = args.leed_out_len
    burnin = args.burnin
    max_iter = args.max_iter
    sample = args.sample

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = AutoModelForMaskedLM.from_pretrained(model_version)

    CLS = "[CLS]"
    SEP = "[SEP]"
    MASK = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    DEVICE = get_device(allow_mps=False)
    model.to(DEVICE)

    ds_book_corpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)[
        "train"
    ]
    ds_book_corpus = ds_book_corpus.select(range(4))
    entry_idxs = []
    bert_sents = []
    for n_sample in tqdm(range(n_samples)):
        random_idx = np.random.randint(len(ds_book_corpus))
        random_entry = ds_book_corpus[random_idx]["text"]
        print(random_entry)
        tokens = tokenizer.tokenize(random_entry)[:5]
        tokens_str = "[CLS]".split()
        tokens_str.extend(tokens)

        for temp in [temperature]:
            bert_sents += generate(
                1,
                mask_id=mask_id,
                sep_id=sep_id,
                seed_text=tokens_str,
                batch_size=batch_size,
                max_len=max_len,
                sample=sample,
                top_k=top_k,
                temperature=temp,
                burnin=burnin,
                max_iter=max_iter,
                cuda=True if DEVICE == torch.device("cuda") else False,
            )
            entry_idxs.append(random_idx)
    out_file = "%s-len%d-burnin%d-topk%d-temp%.3f.txt" % (
        model_version,
        max_len,
        burnin,
        top_k,
        temp,
    )
    write_sents(out_file, bert_sents, should_detokenize=True)
    with open(out_file.replace(".txt", "-idxs.txt"), "w") as out_fh:
        for idx in entry_idxs:
            out_fh.write("%s\n" % idx)
