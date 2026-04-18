import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dummy_dataset_generator import generate_balanced_dataset

data = generate_balanced_dataset(n_samples=20000, length=50)

texts = [x[0] for x in data]
labels = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, shuffle=True
)
pass
class Layer:
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(out_size, in_size) * np.sqrt(2.0 / in_size) #подробнее изучить тему инициализации весов
        self.bias = np.zeros(out_size)

    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.bias
        self.a = np.maximum(0, self.z) #ReLu
        return self.a

    def backward(self, grad_out, lr=0.01):
        dz = grad_out.copy()
        dz[self.z <= 0] = 0
        dW = np.outer(dz, self.x)
        dx = self.W.T @ dz
        self.W -= lr * dW

        db = dz
        self.bias -= lr * db

        return dx

class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)

    def backward(self, pred, target, lr=0.01):
        grad = pred - target

        # идём в обратном порядке по слоям
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def save_weights(self, path="weights.npz"):
        data = {}

        for i, layer in enumerate(self.layers):
            data[f"W_{i}"] = layer.W
            data[f"b_{i}"] = layer.bias

        np.savez(path, **data)

    def load_weights(self, path="weights.npz"):
        data = np.load(path)

        for i, layer in enumerate(self.layers):
            layer.W = data[f"W_{i}"]
            layer.bias = data[f"b_{i}"]

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

model = Model([
        Layer(1550, 256),
        Layer(256, 2)
     ])

num_epochs = 100
num_samples = 1000

def string_to_vector(s):
    char_to_idx = {c: i for i, c in enumerate(list("аеёиоуыэюябвгджзйклмнпрстфхцчшщ"))}
    vec = np.zeros(len(s) * 31, dtype=np.float32)

    for i, ch in enumerate(s):
        idx = char_to_idx.get(ch)
        vec[i * 31 + idx] = 1.0

    return vec

import numpy as np
from tqdm import tqdm

def train(model, X_train, y_train,
          num_epochs=20, lr=0.01,start_step = 0):

    n_train = len(X_train)
    step = start_step

    for epoch in range(1, num_epochs + 1):

        indices = np.random.permutation(n_train)

        correct = 0
        total_loss = 0

        print(f"\nЭпоха {epoch}/{num_epochs}")

        pbar = tqdm(indices, desc="Train", ncols=120)

        for i in pbar:

            step += 1

            x = string_to_vector(X_train[i])
            y = y_train[i]

            pred = model.forward(x)

            target = np.zeros(2, dtype=np.float32)
            target[y] = 1.0

            eps = 1e-12
            loss = -np.log(pred[y] + eps)

            total_loss += loss

            model.backward(pred, target, lr=lr)

            if np.argmax(pred) == y:
                correct += 1

            # ===== REALTIME tqdm update =====
            avg_loss = total_loss / (pbar.n + 1)
            acc = correct / (pbar.n + 1) * 100

            pbar.set_postfix({
                "loss": f"{avg_loss:.10f}",
                "acc": f"{acc:.2f}%"
            })

            if step % 10000 == 0:
                model.save_weights(path=f"step_{step}_weights.npz")

        train_acc = correct / n_train * 100
        if train_acc >= 100:
            lr = 0.00001
        print(f"\nEpoch train accuracy: {train_acc:.2f}%")


def load(model: Model, path):
    model.load_weights(path=path)

load(model, "step_700000_weights.npz")

train(
    model,
    X_train, y_train,
    num_epochs=10000,
    lr=0.01,
    start_step=700000
)


correct_count = 0

with tqdm(range(num_samples), desc="Тестирование", ncols=100, leave=False) as pbar:
    for i in pbar:
        x = string_to_vector(X_test[i])

        pred = model.forward(x)

        if np.argmax(pred) == y_test[i]:
            correct_count += 1

        accuracy = correct_count / (i + 1) * 100
        pbar.set_postfix(accuracy=f"{accuracy:.2f}%")

accuracy = correct_count / num_samples * 100
print(f"Точность на тесте: {accuracy:.2f}%")
