import argparse
import random
import re

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from text_hrm import HRMText

# -------------------
# Dataset sintético
# -------------------

AUTHORS = ["Ana", "Bruno", "Carla"]
DOC_TYPES = ["Oficio", "Peticao", "Relatorio"]
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

def synth_example():
    author = random.choice(AUTHORS)
    doc_t  = random.choice(DOC_TYPES)
    year   = random.choice(YEARS)

    # datas em formatos variados
    dd = random.randint(1,28); mm = random.randint(1,12)
    patterns = [
        f"{dd:02d}/{mm:02d}/{year}",
        f"{year}-{mm:02d}-{dd:02d}",
        f"{dd} de {['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'][mm-1]} de {year}",
    ]
    date_str = random.choice(patterns)

    templates = [
        f"{doc_t} encaminhado por {author} em {date_str} para apreciação da equipe técnica.",
        f"Conforme {date_str}, {author} protocolou {doc_t} no sistema SEI.",
        f"Registro: {doc_t} – autoria {author} – data {date_str}.",
        f"No dia {date_str}, {author} finalizou o {doc_t} com anexos.",
        f"{author} elaborou o {doc_t} em {date_str}, conforme despacho.",
    ]
    text = random.choice(templates)
    return text, AUTHORS.index(author), DOC_TYPES.index(doc_t), YEARS.index(year)

def build_vocab(samples, min_freq=1):
    from collections import Counter
    counter = Counter()
    for s in samples:
        toks = tokenize(s)
        counter.update(toks)
    # ids: 0=PAD
    vocab = {"<PAD>":0}
    for tok, cnt in counter.items():
        if cnt >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def tokenize(s):
    # tokenização simplificada por palavras e números
    s = s.lower()
    # separa pontuação
    s = re.sub(r"([.,;:()–-])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def encode(s, vocab, max_len=64):
    toks = tokenize(s)
    ids = [vocab.get(t, vocab.setdefault("<UNK>", len(vocab))) for t in toks]
    ids = ids[:max_len]
    pad_id = vocab["<PAD>"]
    if len(ids) < max_len:
        ids += [pad_id]*(max_len - len(ids))
    return ids, min(len(toks), max_len)

def batchify(batch, vocab, max_len=64, device="cpu"):
    xs, a_idx, t_idx, y_idx = [], [], [], []
    lens = []
    for txt, a, t, y in batch:
        ids, L = encode(txt, vocab, max_len=max_len)
        xs.append(ids); lens.append(L)
        a_idx.append(a); t_idx.append(t); y_idx.append(y)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    lengths = torch.tensor(lens, dtype=torch.long, device=device)
    a = torch.tensor(a_idx, dtype=torch.long, device=device)
    t = torch.tensor(t_idx, dtype=torch.long, device=device)
    y = torch.tensor(y_idx, dtype=torch.long, device=device)
    return x, lengths, a, t, y

def make_dataset(n):
    data = [synth_example() for _ in range(n)]
    return data

# -------------------
# Train / Eval
# -------------------

def accuracy(logits, target):
    pred = logits.argmax(dim=-1)
    return (pred == target).float().mean().item()

def triplet_acc(la, lt, ly, ta, tt, ty):
    pa, pt, py = la.argmax(-1), lt.argmax(-1), ly.argmax(-1)
    return ((pa==ta) & (pt==tt) & (py==ty)).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    # Data
    train_data = make_dataset(args.train)
    val_data = make_dataset(args.val)

    # Vocab
    vocab = build_vocab([t for t,_,_,_ in train_data] + [t for t,_,_,_ in val_data])
    pad_id = vocab["<PAD>"]
    n_auth, n_type, n_year = len(AUTHORS), len(DOC_TYPES), len(YEARS)

    # Model
    model = HRMText(vocab_size=len(vocab),
                    d_model=args.d_model,
                    steps=args.steps, cycles=args.cycles,
                    n_authors=n_auth, n_types=n_type, n_years=n_year,
                    pad_id=pad_id).to(device)

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    def run_epoch(data, train=True):
        if train: model.train()
        else: model.eval()

        total_loss = 0.0
        A=T=Y=Trip=0.0
        pbar = tqdm(range(0, len(data), 32), desc="train" if train else "val")
        for i in pbar:
            batch = data[i:i+32]
            x, lengths, a, t, y = batchify(batch, vocab, max_len=args.max_len, device=device)

            with torch.set_grad_enabled(train):
                la, lt, ly = model(x, lengths)
                loss = ce(la, a) + ce(lt, t) + ce(ly, y)

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            total_loss += loss.item()
            A += accuracy(la, a)
            T += accuracy(lt, t)
            Y += accuracy(ly, y)
            Trip += triplet_acc(la, lt, ly, a, t, y)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_batches = max(1, len(data)//32 + (1 if len(data)%32 else 0))
        return (total_loss/n_batches, A/n_batches, T/n_batches, Y/n_batches, Trip/n_batches)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = run_epoch(train_data, train=True)
        va = run_epoch(val_data, train=False)
        print(f"[train] loss={tr[0]:.3f}  acc_author={tr[1]:.3f}  acc_type={tr[2]:.3f}  acc_year={tr[3]:.3f}  triplet={tr[4]:.3f}")
        print(f"[val]   loss={va[0]:.3f}  acc_author={va[1]:.3f}  acc_type={va[2]:.3f}  acc_year={va[3]:.3f}  triplet={va[4]:.3f}")

    torch.save({"model": model.state_dict(), "vocab": vocab}, "hrm_text.pt")
    print("Modelo salvo em hrm_text.pt")

if __name__ == "__main__":
    main()
