"""Single-turn assistant-only labels; reject rather than truncate long answers."""
import torch

from .runtime import prompt_ids


def encode_sample(row, tokenizer, max_length):
    messages = row["conversations"]
    if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("v1 accepts exactly user/assistant single-turn samples")
    prefix = prompt_ids(tokenizer, messages[:1])
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, open_thinking=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("training template does not share the non-thinking generation prefix")
    if len(full) > max_length:
        raise ValueError(f"sample {row.get('id')} has {len(full)} tokens > max_length={max_length}; no truncation allowed")
    if len(full) <= len(prefix):
        raise ValueError("sample has no supervised answer tokens")
    labels = [-100] * len(prefix) + full[len(prefix):]
    return {"input_ids": full, "labels": labels}


def collate(rows, pad_id):
    size = max(len(r["input_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor([r["input_ids"] + [pad_id] * (size - len(r["input_ids"])) for r in rows]),
        "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (size - len(r["input_ids"])) for r in rows]),
        "labels": torch.tensor([r["labels"] + [-100] * (size - len(r["labels"])) for r in rows]),
    }


def target_tokens(batch):
    return int((batch["labels"][:, 1:] != -100).sum())
