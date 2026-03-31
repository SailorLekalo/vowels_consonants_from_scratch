import random
import csv

VOWELS = list("аеёиоуыэюя")
CONSONANTS = list("бвгджзйклмнпрстфхцчшщ")

def generate_controlled_string(length=512, target_label=1):
    """
    target_label = 1 → больше гласных
    target_label = 0 → больше согласных
    """
    # Минимальный перевес: +1
    if target_label == 1:
        n_vowels = length // 2 + 1
    else:
        n_vowels = length // 2 - 1

    n_consonants = length - n_vowels

    chars = (
        [random.choice(VOWELS) for _ in range(n_vowels)] +
        [random.choice(CONSONANTS) for _ in range(n_consonants)]
    )

    random.shuffle(chars)
    return ''.join(chars)

def generate_balanced_dataset(n_samples, length=512):
    assert n_samples % 2 == 0, "Для идеального баланса нужно чётное число сэмплов"

    data = []

    # Половина — класс 1
    for _ in range(n_samples // 2):
        s = generate_controlled_string(length, target_label=1)
        data.append((s, 1))

    # Половина — класс 0
    for _ in range(n_samples // 2):
        s = generate_controlled_string(length, target_label=0)
        data.append((s, 0))

    random.shuffle(data)
    return data

def save_to_csv(data, filename="dataset.csv"):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(data)

if __name__ == "__main__":
    dataset = generate_balanced_dataset(n_samples=1000, length=100)
    save_to_csv(dataset)