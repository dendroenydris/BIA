"""Chunked CLIP prompt encoding for prompts longer than the 77-position SD 2.1 context."""


def chunk_ids(tokenizer, text, max_length=77):
    """Split a prompt into BOS + up to 75 tokens + EOS chunks, padded to max_length.

    The first chunk equals standard right-truncated tokenization, so chunk 1 alone
    reproduces the historical truncated input and all chunks together encode the full text.
    """
    ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    body = max_length - 2
    pieces = [ids[i:i + body] for i in range(0, len(ids), body)] or [[]]
    return [[tokenizer.bos_token_id] + p + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (body - len(p))
            for p in pieces]


def encode(pipe, text, truncate):
    """Return (prompt_embeds, negative_prompt_embeds) for a full or truncated prompt."""
    import torch
    device = pipe.text_encoder.device
    chunks = chunk_ids(pipe.tokenizer, text)
    if truncate:
        chunks = chunks[:1]
    with torch.no_grad():
        enc = lambda c: pipe.text_encoder(torch.tensor([c], device=device))[0]
        prompt = torch.cat([enc(c) for c in chunks], dim=1)
        negative = torch.cat([enc(chunk_ids(pipe.tokenizer, '')[0])] * len(chunks), dim=1)
    return prompt, negative
