import torch


def calculate_perplexity(
    sentence: str, model, tokenizer, device: torch.device
) -> float:
    """
    Calculate the perplexity of a given sentence using a language model.

    Perplexity is a measurement of how well a probability model predicts a sample.
    It is commonly used in natural language processing tasks to evaluate language models.

    Args:
        sentence (str): The input sentence for which to calculate perplexity.
        model: The language model to evaluate. It should be a model compatible with the input format.
        tokenizer: The tokenizer used to encode the input sentence into token IDs.
        device (torch.device): The device (CPU or GPU) on which to perform the calculations.

    Returns:
        float: The calculated perplexity of the input sentence. A lower perplexity indicates
               that the model is better at predicting the sentence.

    Notes:
        - The function assumes that the model has already been loaded and is in evaluation mode.
        - The model's outputs are expected to be in the format where the first output is the loss.
        - The function does not handle cases where the model or tokenizer is not compatible
          with the input sentence; appropriate checks should be performed before calling this function.
        - Ensure that the model and tokenizer are initialized and moved to the correct device before
          invoking this function.

    Example:
        >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model.to(device)
        >>> sentence = "The quick brown fox jumps over the lazy dog."
        >>> perplexity = calculate_perplexity(sentence, model, tokenizer, device)
        >>> print(f"Perplexity: {perplexity.item()}")
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return torch.exp(loss)
