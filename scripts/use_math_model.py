"""Use a verified local LoRA adapter, or export a standalone merged HF model."""
import argparse
from pathlib import Path

import torch

from mathlab.data import SUFFIX
from mathlab.io import sha256, write_json
from mathlab.lora import merge
from mathlab.runtime import (load_adapter, load_model, model_identity, prompt_ids,
                             resolve_model, select_device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["ask", "export"])
    parser.add_argument("question", nargs="?", default="计算 12 + 7。")
    parser.add_argument("--adapter", default="artifacts/release/adapter")
    parser.add_argument("--model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="artifacts/release/merged")
    args = parser.parse_args()
    path, device = resolve_model(args.model), select_device(args.device)
    identity = model_identity(path)
    model, tokenizer = load_model(path, device)
    metadata = load_adapter(model, args.adapter, identity)
    model.eval()
    question = args.question if args.question.endswith(SUFFIX) else args.question + SUFFIX
    ids = prompt_ids(tokenizer, [{"role": "user", "content": question}])
    inputs = torch.tensor([ids], device=device)
    mask = torch.ones_like(inputs)
    if args.action == "ask":
        if len(ids) + 256 > model.config.max_position_embeddings:
            raise ValueError("question exceeds context budget")
        output = model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=256,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
        print(tokenizer.decode(output[0, len(ids):], skip_special_tokens=True))
        return
    destination = Path(args.out)
    if destination.exists():
        raise ValueError("export directory exists; choose a new destination")
    before = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits.detach().cpu()
    merge(model)
    after = model(input_ids=inputs, attention_mask=mask, use_cache=False).logits.detach().cpu()
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
    destination.mkdir(parents=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    reloaded, _ = load_model(destination, device)
    reloaded.eval()
    disk = reloaded(input_ids=inputs, attention_mask=mask, use_cache=False).logits.detach().cpu()
    torch.testing.assert_close(disk, after, rtol=0, atol=0)
    report = dict(base_files=identity, adapter=str(Path(args.adapter).resolve()),
                  adapter_sha256=metadata["weights_sha256"], merged_files=model_identity(destination),
                  test_question=question, merge_max_absolute_logit_difference=(before - after).abs().max().item(),
                  reloaded_logits_exact_equal=torch.equal(after, disk), dtype="float32", device=str(device),
                  tolerance=dict(rtol=1e-4, atol=1e-4), script_sha256=sha256(__file__),
                  limitation="single prompt numerical export check; capability measured in campaign test runs")
    write_json(destination / "export_verification.json", report)
    print(f"Exported verified model: {destination.resolve()}")


if __name__ == "__main__":
    main()
