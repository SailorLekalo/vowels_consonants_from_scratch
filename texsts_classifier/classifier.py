import math

import numpy as np
import csv

from texsts_classifier.dummy_dataset_generator import generate_balanced_dataset

def load_dataset(path):
    texts = []
    labels = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))

    return texts, np.array(labels)

texts, y = load_dataset("dataset.csv")

def build_vocab(texts): #словарь "символ - индекс"
    chars = set()
    for t in texts:
        chars.update(t)

    chars = sorted(list(chars))
    char2idx = {c: i for i, c in enumerate(chars)}

    return char2idx

char2idx = build_vocab(texts)
vocab_size = len(char2idx)

embed_dim = 1 #каждому символу будет соответствовать одно число

#создаём эмбеддинг -- матрицу "индекс в словаре -> одно число (вес). Чисел могло бы быть несколько, но нам достаточно одно
embedding = np.random.randn(vocab_size,embed_dim) * 0.01

def sigmoid(x): #не ReLu, потому что нам важны отрицательные значения -- согласные "снижают" вероятность, в т.ч. в минуса.
    return 1 / (1 + math.exp(-x))

def forward(text, char2idx, embedding):
    total = 0

    for c in text:
        total += embedding[char2idx[c]][0]

    prob = sigmoid(total)
    return prob

def backward(text, char2idx, embedding, prob, y_true, lr=0.01):
    grad = prob - y_true
    for c in text:
        idx = char2idx[c]
        embedding[idx][0] -= lr * grad

def save_model(path, embedding, char2idx):
    np.savez(
        path,
        embedding=embedding,
        char2idx=char2idx
    )

def load_model(path):
    data = np.load(path, allow_pickle=True)

    embedding = data["embedding"]
    char2idx = data["char2idx"].item()  # важно: это dict

    return embedding, char2idx

def train(texts, y, char2idx, embedding, epochs=5, lr=0.01):
    for epoch in range(1, epochs + 1):
        correct = 0

        for i in range(len(texts)):
            text = texts[i]
            y_true = y[i]

            # forward
            prob = forward(text, char2idx, embedding)

            # accuracy
            pred = 1 if prob > 0.5 else 0
            if pred == y_true:
                correct += 1

            # backward
            backward(text, char2idx, embedding, prob, y_true, lr)

        acc = correct / len(texts) * 100

        print(f"Epoch {epoch}: acc={acc:.2f}%")
    save_model("model.npz", embedding, char2idx)
#train(texts, y, char2idx, embedding, epochs=10, lr=0.01)

def test_model():
    a = 0
    num_test=10000
    embedding, char2idx = load_model("model.npz")
    for i in generate_balanced_dataset(num_test):
        prob = forward(i[0],char2idx, embedding)
        pred = 1 if prob > 0.5 else 0
        if pred == i[1]:
            a+=1
        else:
            a-=1
    print(a/num_test)
test_model()
