#!/usr/bin/env python

from torch.utils.data import Dataset, DataLoader
import re

from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

import os

import numpy as np

import json

from functools import partial

from torch.utils.data import DataLoader

import time

import argparse

import psutil

import urllib.request

import shlex
import subprocess
import readline

import configparser
import getpass


GPT_CONFIG_AI = {
    "vocab_size": 50257,    
    "context_length": 1024, 
    "emb_dim": 1024,        
    "n_heads": 16,          
    "n_layers": 24,         
    "drop_rate_emb": 0.1,   
    "drop_rate_att": 0.1,   
    "drop_rate_shc": 0.1,   
    "qkv_bias": True       
}



def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # copy

    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[List[str]] = None,
    ):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        default_specials = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
        self.special_tokens = special_tokens or default_specials

        max_id = max(self.encoder.values())
        for tok in self.special_tokens:
            if tok not in self.encoder:
                max_id += 1
                self.encoder[tok] = max_id
                self.decoder[max_id] = tok

        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        self.special_regex = re.compile(f"({pattern})")

        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        self.cache: Dict[str, str] = {}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    def get_pairs(self, word: List[str]) -> set:
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = list(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            min_pair = min(
                (p for p in pairs if p in self.bpe_ranks),
                key=lambda p: self.bpe_ranks[p],
                default=None,
            )
            if min_pair is None:
                break
            first, second = min_pair
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1

            word = new_word
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        out = " ".join(word)
        self.cache[token] = out
        return out

    def _split_on_special(self, text: str) -> List[str]:
        parts = self.special_regex.split(text)
        return [p for p in parts if p]

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[str] = []

        for segment in self._split_on_special(text):
            if segment in self.special_tokens:
                bpe_tokens.append(segment)
            else:
                data = segment.encode("utf-8")
                unicode_str = "".join(self.byte_encoder[b] for b in data)
                for sub in self.bpe(unicode_str).split(" "):
                    bpe_tokens.append(sub)

        return [self.encoder[token] for token in bpe_tokens]

    def decode(self, token_ids: List[int]) -> str:
        out_text = ""
        for idx in token_ids:
            token = self.decoder.get(idx, "")
            if token in self.special_tokens:
                out_text += token
            else:
                byte_seq = bytes(self.byte_decoder[ch] for ch in token)
                out_text += byte_seq.decode("utf-8", errors="replace")
        return out_text


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_att"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shc"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), 
                GELU(), 
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text) 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size,
    temperature=0.0, top_k=None, eos_id=None):

    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(prompt, model="llama3.1:8b", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data


def parse_options():
    # COMMAND LINE OPTIONS LOGIC #
    parser = argparse.ArgumentParser(description="Personal AI assistant.")
    parser.add_argument("--model", type=str, help="Import GPT2 compatible model")
    args = parser.parse_args()

    arg_convert = None
    arg_pretrain = None
    arg_model = None
    arg_finetune = None
    arg_evaluate = None
    arg_savefile = None
    arg_profile = None

    if args.model != None and os.path.exists(args.model):
        arg_model = args.model
    else:
        arg_model = "model.pth"

    return arg_convert, arg_pretrain, arg_model, arg_finetune, arg_evaluate, arg_savefile, arg_profile


def instantiate_model(device, arg_model):
    gpt = GPTModel(GPT_CONFIG_AI)                         
    gpt.to(device)                                          # Correct position #
    checkpoint = torch.load(arg_model, map_location=device) # Correct position #
    gpt.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    gpt.train();
    gpt.eval()
        
    print("LOADED LLM SUCCESSFULLY.\n")
    return gpt


def read_config(config_file = "~/.navirc"):
    config = configparser.ConfigParser()
    config.read(config_file)

    # ...or via get() with defaults
    finetuning_log = config.get("settings", "FINETUNIG_LOG", fallback="/tmp/navi_finetuning.json")
    comment_ai_model = config.get("settings", "COMMENT_AI_MODEL", fallback="llama3.1:8b")

    return finetuning_log, comment_ai_model


def copy_history():
    copy_of_history = []
    for idx in range(1,readline.get_current_history_length()+1):
        copy_of_history.append(readline.get_history_item(idx))
    return copy_of_history


def empty_history():
    readline.clear_history()


def restore_history(copy_of_history):
    if len(copy_of_history) > 0:
        for idx in range(0, len(copy_of_history)):
            entry = copy_of_history[idx]
            if entry != None:
                readline.add_history(entry)


def remove_last_from_history():
    last_index = readline.get_current_history_length() - 1
    if last_index >= 0:
        readline.remove_history_item(last_index)


def prefill_input(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.set_startup_hook(None)
    readline.set_startup_hook(hook)  # clear the hook afterwards
    return input(prompt)


def write_finetuning_file(finetuning_data, comment, line):
    with open(finetuning_data, "a", encoding="utf-8") as file:
        data = {"instruction": comment, "input": "", "output": line}
        json.dump(data, file, ensure_ascii=True, indent=2)
        file.write("\n")


def run_model(gpt, device, tokenizer):
    print("NAVIGATOR SHELL started.\n")
    finetuning_log, comment_ai_model = read_config()

    commands_history = []
    comments_history = []

    copy_of_commands = []
    copy_of_comments = []

    repeat_command = False
    skip_command = False

    while True:
        try:
            cwd = os.getcwd()
            usr = getpass.getuser()

            if not repeat_command:
                restore_history(copy_of_comments)
                comment = input(f"[AIPrompt]{usr}:{cwd}$ ").strip()
                if len(comment) > 1 and comment[len(comment)-1] != '.':
                    comment += '.'
                
                candidate_line = ''
                if comment != '' and comment not in ["exit", "exit.", "bye", "bye."]:
                    user_text = (
                        f"Below is an instruction that describes a task. "
                        f"Write a response that appropriately completes the request."
                        f"\n\n### Instruction:\n{comment}"
                    )
                    output_offset = len(user_text)

                    token_ids = generate(
                        model=gpt,
                        idx=text_to_token_ids(user_text, tokenizer).to(device),
                        max_new_tokens=256,
                        context_size=GPT_CONFIG_AI["context_length"],
                        eos_id=50256
                    )
                    output_text = token_ids_to_text(token_ids, tokenizer)
                    candidate_line = output_text[output_offset:].replace("### Response:", "").strip()
                    skip_command = False
                elif comment in ["exit", "exit.", "bye", "bye."]:
                    line = 'exit'
                    skip_command = True

                copy_of_comments = copy_history()
                empty_history()

            if not skip_command:
                restore_history(copy_of_commands)

                line = prefill_input(f"[CoMmanD ]{usr}:{cwd}$ ", candidate_line).strip()
                if line != candidate_line and comment not in (".", ""):
                    write_finetuning_file(finetuning_log, comment, line) # FEATURE #
                if not line:
                    repeat_command = True
                    remove_last_from_history()
                    continue
                else:
                    repeat_command = False
                copy_of_commands = copy_history()
                empty_history()

                if comment.strip() == '' and line not in ["exit", "exit.", "bye", "bye."]:
                    ollama_running = check_if_running("ollama")
                    if not ollama_running:
                        comment_warning = "Comment was empty. Last chance to insert one."
                        comment = prefill_input(f"[COMment ]{usr}:{cwd}$ ", comment_warning)

                    else:
                        #model = "llama3.1:8b"
                        model = comment_ai_model
                        candidate_comment = query_model(f"Generate a short one-line description for the following Linux shell command. Type only the description, don't use new line or quotes characters:\n{line}", model)
                        comment = prefill_input(f"[COMment ]{usr}:{cwd}$ ", candidate_comment)
                        write_finetuning_file(finetuning_log, comment, line) # FEATURE #

                commands_history.append(line)
                comments_history.append(comment)

            args = shlex.split(line)

            # Show combined history
            if args[0] == "history" and len(args) == 1:
                for i, (cmd, com) in enumerate(zip(commands_history, comments_history), 1):
                    print(f"{i}. {cmd}    # {com}")
                continue

            # Show only commands history
            if args[0] == "history" and len(args) == 2 and args[1] == "commands":
                for i, cmd in enumerate(commands_history, 1):
                    print(f"{i}. {cmd}")
                continue

            # Show only comments history
            if args[0] == "history" and len(args) == 2 and args[1] == "comments":
                for i, com in enumerate(comments_history, 1):
                    print(f"{i}. {com}")
                continue

            # Execute command number
            if args[0] == "history" and len(args) == 2 and type(int(args[1])) == int:
                line = commands_history[int(args[1])-1]

            # Change directory
            if args[0] == "cd":
                target = args[1] if len(args) > 1 else os.path.expanduser("~")
                try:
                    os.chdir(target)
                except FileNotFoundError:
                    print(f"cd: no such file or directory: {target}")
                continue

            # Exit shell
            if args[0] in ["exit", "exit.", "bye", "bye."]:
                break

            try:
                proc = subprocess.Popen(line, shell=True, executable="/bin/bash",
                                        stdin=None, stdout=None, stderr=None)
                proc.communicate()
            except FileNotFoundError:
                print("shell not found")

        except KeyboardInterrupt:
            print()   # handle Ctrl+C
        except EOFError:
            print()   # handle Ctrl+D
            print("NAVIGATOR SHELL stopped.\n")
            break



def main():
    arg_convert, arg_pretrain, arg_model, arg_finetune, arg_evaluate, arg_savefile, arg_profile = parse_options()

    vocab_file  = "vocab.json"
    merges_file = "merges.txt"
    tokenizer = BPETokenizer(vocab_file, merges_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(arg_model):
        print("Error: missing default 'model.pth' model file. Please pretrain from scratch.")
        exit(15)

    gpt = instantiate_model(device, arg_model)

    run_model(gpt, device, tokenizer)



if __name__ == '__main__':
    main()

