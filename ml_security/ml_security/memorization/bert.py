import argparse
import math
import time

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def detokenize(sent):
    """Roughly detokenizes (mainly undoes wordpiece)"""
    new_sent = []
    for i, tok in enumerate(sent):
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

    # Generation modes as functions


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """Get initial sentence by padding seed_text with either masks or random words to max_len"""
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    # if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

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
    """Generate for one random position at a timestep

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
            print("iter", ii + 1, " ".join(for_print))

    return untokenize_batch(batch)


def parallel_generation(
    seed_text,
    max_len=15,
    top_k=0,
    temperature=None,
    max_iter=300,
    sample=True,
    cuda=False,
    print_every=10,
    verbose=True,
    batch_size=2,
):
    """Generate for all positions at a time step"""
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        for kk in range(max_len):
            idxs = generate_step(
                out,
                gen_idx=seed_len + kk,
                top_k=top_k,
                temperature=temperature,
                sample=sample,
            )
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = idxs[jj]

        if verbose and np.mod(ii, print_every) == 0:
            print("iter", ii + 1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))

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
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
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
        # batch = parallel_generation(seed_text, max_len=max_len, top_k=top_k, temperature=temperature, sample=sample, max_iter=max_iter)

        if (batch_n + 1) % print_every == 0:
            print(
                "Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time)
            )
            start_time = time.time()

        sentences += batch
    return sentences


# Utility functions
def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def read_sents(in_file, should_detokenize=False):
    sents = [sent.strip().split() for sent in open(in_file).readlines()]
    if should_detokenize:
        sents = [detokenize(sent) for sent in sents]
    return sents


def write_sents(out_file, sents, should_detokenize=False):
    with open(out_file, "w") as out_fh:
        for sent in sents:
            sent = detokenize(sent[1:-1]) if should_detokenize else sent
            out_fh.write("%s\n" % " ".join(sent))


###


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


####

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

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

    from datasets import load_dataset

    ds_book_corpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)
    bert_sents = []
    for n_sample in range(n_samples):
        random_entry = ds_book_corpus["train"][
            np.random.randint(len(ds_book_corpus["train"]))
        ]["text"]
        # use the tokenizer and grab the first 5 tokens from the entry
        tokens = tokenizer.tokenize(random_entry)[:5]
        tokens_str = "[CLS] " + " ".join(tokens)

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
                cuda=False,
            )
    out_file = "%s-len%d-burnin%d-topk%d-temp%.3f.txt" % (
        model_version,
        max_len,
        burnin,
        top_k,
        temp,
    )
    write_sents(out_file, bert_sents, should_detokenize=True)
