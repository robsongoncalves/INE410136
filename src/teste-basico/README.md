# Mini-tutorial: HRM-style (sem CoT) para **extração de metadados de texto**

Este exemplo mostra como adaptar o estilo HRM (raciocínio **latente**, sem Chain-of-Thought textual)
para **classificar metadados** a partir de textos curtos (toy dataset). O modelo não imprime passos
intermediários; ele somente **emite as classes finais**: `autor`, `tipo_documento`, `ano`.

> Objetivo didático: demonstrar a ideia em NLP, usando PyTorch puro, tokenização simples e
> dois módulos recorrentes (H e L) com iterações internas.

---

## 🚀 Como rodar

```bash
pip install -r requirements.txt
python train_text_metadata.py --epochs 5 --train 3000 --val 600 --cycles 3 --steps 3
```

Parâmetros úteis:
- `--epochs`: épocas de treino (default 5)
- `--train`, `--val`: nº de exemplos sintéticos de treino/val (default 3000/600)
- `--cycles`: nº de ciclos (H)
- `--steps`: nº de passos do módulo L por ciclo (L)
- `--device`: `cpu` ou `cuda`

Saída: accuracy por campo (autor, tipo, ano) e **triplet-accuracy** (acertar os 3 juntos).

---

## 🧠 Ideia do HRM-style (texto)

- **Embedding** de tokens (vocabulário gerado do dataset sintético).
- **Módulo L (rápido)** roda `T` passos por ciclo ajustando representações locais.
- **Módulo H (lento)** atualiza ao final de cada ciclo e injeta contexto global.
- **Pooling**: usamos o vetor do token `[CLS]` do H para prever as 3 cabeças de saída.
- **Sem CoT**: o modelo faz as iterações **internamente** e retorna só as classes finais.

---

## 📎 Observações

- É um *toy example* — simples e legível. Para dados reais, troque a tokenização,
  aumente embedding/d_model e ajuste o gerador de dados.
- Você pode portar os módulos H/L para GRU/MLP recorrente facilmente.

Bom estudo! 🙂
