import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


config = {
    "n_layer": 4,
    "n_embd": 256,
    "n_head": 4,
}

vocab_size = 50257
max_len = 128
batch_size = 32
learning_rate = 1e-3
epochs = 5

# Названия файлов
text_file = "Учень.txt"
weights_file = "Учень.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, config["n_embd"]),
            wpe=nn.Embedding(max_len, config["n_embd"]),
            h=nn.ModuleList([Block() for _ in range(config["n_layer"])]),
            ln_f=nn.LayerNorm(config["n_embd"]),
        ))
        self.lm_head = nn.Linear(config["n_embd"], vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

model = MiniGPT().to(device)
enc = tiktoken.get_encoding("gpt2")

def train_model():
    if not os.path.exists(text_file):
        print(f"! Файл '{text_file}' ")
        sample_data = "учень. " * 3000
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(sample_data)

    print(f"52ptp'{text_file}'...")
    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    data = torch.tensor(enc.encode(raw_text), dtype=torch.long)
    print(f"сжат в {len(data):,} токенов.")

    def get_batch():
        ix = torch.randint(len(data) - max_len, (batch_size,))
        x = torch.stack([data[i:i + max_len] for i in ix])
        y = torch.stack([data[i + 1:i + max_len + 1] for i in ix])
        return x.to(device), y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    iterations = max(100, len(data) // (batch_size * max_len) * epochs)

    print(f"\n учень учится: {device.upper()}")
    print(f" iter: {iterations}.")

    model.train()
    for step in range(iterations):
        xb, yb = get_batch()
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == iterations - 1:
            print(f"Шаг {step}/{iterations} |(Loss): {loss.item():.4f}")

    torch.save(model.state_dict(), weights_file)
    print(f"\n Учень умен: {weights_file}")


def run_chat():
    print(f"\n веса из '{weights_file}'...")
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    print("ready")

    system_instruction = "Instruction: You are a smart AI assistant. Answer the user question clearly.Ты помощник, отвечай на вопросы.\n"

    while True:
        user_input = input("\n:")
        if user_input.lower() in ["выход", "exit", "quit"]:
            print("67")
            break

        if not user_input.strip():
            continue

        full_prompt = (f"{system_instruction}"
                       f": {user_input}\nAI:")

        context_tokens = torch.tensor([enc.encode(full_prompt)], dtype=torch.long, device=device)


        with torch.no_grad():
            generated_tokens = model.generate(context_tokens, max_new_tokens=500, temperature=0.4)


        full_text = enc.decode(generated_tokens.tolist()[0])


        bot_response = full_text[len(full_prompt):]


        if "User:" in bot_response:
            bot_response = bot_response.split("User:")[0]

        print(f"Нейросеть: {bot_response.strip()}")


if __name__ == "__main__":
    if not os.path.exists(weights_file):
        print("нет весов")
        train_model()

    run_chat()
