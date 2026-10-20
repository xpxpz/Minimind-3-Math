import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_hf_model(tmp_path):
    # Character tokenizer: no downloads, and no BPE ambiguity at prefix boundaries.
    specials = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
    alphabet = sorted(set("userassistantthink答案：数值只需给出最终结果格式为非整数请用最简分不要带单位计算表达式的值是多少盒子里有个球又放入现在支铅笔掉还剩每装共把米绳平均成段长天匀速修完千道路水箱原升流出上午售张票下两次一需要元买?？，。；“”0123456789+-*/() \n<>"))
    vocab = {s: i for i, s in enumerate(specials + alphabet)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    raw.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>",
                                        eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tokenizer.chat_template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' }}"
                               "{% if m['role'] == 'assistant' %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
                               "{{ m['content'] + '<|im_end|>\n' }}{% endfor %}"
                               "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}")
    path = tmp_path / "tiny-model"
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                      num_key_value_heads=1, head_dim=8, intermediate_size=32,
                                      vocab_size=len(tokenizer), max_position_embeddings=512,
                                      tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path, tokenizer
