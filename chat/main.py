import numpy as np
import os
import csv
import logging
from itertools import product

# Configuração do logger
type(logging)  # prevent unused import warning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======= Parâmetros ========
X_NPY_PATH = 'X.npy'          # caminho do arquivo X.npy (shape: N,10,12,1)
Y_NPY_PATH = 'Y_classe.npy'   # caminho do arquivo Y_classe.npy (shape: N,26)
OUTDIR     = 'output'         # diretório de saída
TEST_SIZE  = 130              # número de amostras no conjunto de teste (últimas 130)
SEED       = 42              # semente aleatória
CV_FOLDS   = 5               # número de folds para cross-validation

# Parâmetros para grid search
GRID_HIDDEN   = [25, 50, 75, 100]
GRID_LR       = [0.005, 0.01, 0.05, 0.1]
GRID_EPOCHS   = [50, 100, 200]
GRID_PATIENCE = [3, 5, 10]
# ============================

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 learning_rate=0.01, seed=None, patience=None, max_epochs=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs
        # pesos com bias
        self.W1 = np.random.randn(n_hidden, n_inputs + 1) * 0.1
        self.W2 = np.random.randn(n_outputs, n_hidden + 1) * 0.1

    @staticmethod
    def _activate_hidden(x):
        return np.tanh(x)

    @staticmethod
    def _activate_hidden_derivative(y):
        return 1.0 - y**2

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def feedforward(self, X):
        xb = np.append(X, 1.0)
        self.z = self._activate_hidden(self.W1.dot(xb))
        zb = np.append(self.z, 1.0)
        logits = self.W2.dot(zb)
        self.y = self._softmax(logits)
        return self.y

    def backprop_step(self, X, T):
        Y = self.feedforward(X)
        # cross-entropy gradient: Y - T
        delta_out = Y - T
        # hidden delta
        delta_hidden = self._activate_hidden_derivative(self.z) * (self.W2[:, :-1].T.dot(delta_out))
        # atualiza W2 e W1
        self.W2 -= self.lr * np.outer(delta_out, np.append(self.z, 1.0))
        self.W1 -= self.lr * np.outer(delta_hidden, np.append(X, 1.0))
        # retorna cross-entropy loss
        return -np.sum(T * np.log(Y + 1e-15))

    def train(self, X_train, Y_train, X_val=None, Y_val=None):
        best_val = float('inf')
        epochs_no_improve = 0
        for epoch in range(1, self.max_epochs + 1):
            train_loss = 0.0
            for x, t in zip(X_train, Y_train):
                train_loss += self.backprop_step(x, t)
            val_loss = None
            if X_val is not None:
                val_loss = sum(-np.sum(t * np.log(self.feedforward(x) + 1e-15))
                               for x, t in zip(X_val, Y_val))
                if self.patience and val_loss < best_val:
                    best_val, epochs_no_improve = val_loss, 0
                    bestW1, bestW2 = self.W1.copy(), self.W2.copy()
                else:
                    epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    self.W1, self.W2 = bestW1, bestW2
                    break
            logging.info(f"Epoch {epoch}/{self.max_epochs} - train_loss: {train_loss:.4f}" + (f", val_loss: {val_loss:.4f}" if val_loss is not None else ''))
        return train_loss, val_loss

    def predict(self, X):
        out = self.feedforward(X)
        # retorna one-hot com 1 na maior probabilidade
        vec = np.zeros_like(out)
        vec[np.argmax(out)] = 1
        return vec

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

# carregamento e pré-processamento
def load_data():
    X = np.load(X_NPY_PATH)
    Y = np.load(Y_NPY_PATH)
    N = X.shape[0]
    X = X.reshape(N, -1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # assume Y já one-hot de 0/1; converte para floats
    Y = Y.astype(float)
    return X, Y

# split determinístico: últimos TEST_SIZE para teste

def train_test_split(X, Y, test_size):
    X_test = X[-test_size:]
    Y_test = Y[-test_size:]
    X_train = X[:-test_size]
    Y_train = Y[:-test_size]
    return X_train, Y_train, X_test, Y_test

# cross-validation manual

def cross_validate(X, Y, params):
    N = X.shape[0]
    indices = np.arange(N)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    fold_sizes = (N // CV_FOLDS) * np.ones(CV_FOLDS, int)
    fold_sizes[:N % CV_FOLDS] += 1
    current = 0
    scores = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        model = MLP(X.shape[1], params['hidden'], Y.shape[1],
                    params['lr'], SEED, params['patience'], params['epochs'])
        model.train(X_tr, Y_tr, X_val, Y_val)
        acc = np.mean((model.predict_batch(X_val) == Y_val).all(axis=1))
        scores.append(acc)
        current = stop
    return np.mean(scores)

# grid search com cross-validation
def grid_search(X, Y):
    results = []
    for hidden, lr, epochs, pat in product(GRID_HIDDEN, GRID_LR, GRID_EPOCHS, GRID_PATIENCE):
        params = {'hidden':hidden, 'lr':lr, 'epochs':epochs, 'patience':pat}
        cv_score = cross_validate(X, Y, params)
        results.append({**params, 'cv_acc':cv_score})
    os.makedirs(OUTDIR, exist_ok=True)
    keys = results[0].keys()
    with open(os.path.join(OUTDIR, 'grid_search.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(results)
    best = max(results, key=lambda x: x['cv_acc'])
    logging.info(f"Best CV config: {best}")
    return best

# avaliação e salvamento da matriz de confusão e acurácia
def evaluate_and_save(model, X_test, Y_test, prefix):
    preds = model.predict_batch(X_test)
    n_out = Y_test.shape[1]
    cm = np.zeros((n_out, n_out), int)
    for true_v, pred_v in zip(Y_test, preds):
        i = np.where(true_v == 1)[0][0]
        j = np.where(pred_v == 1)[0][0]
        cm[i, j] += 1
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    np.savetxt(prefix + '_confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    acc = np.mean((preds == Y_test).all(axis=1))
    with open(prefix + '_accuracy.txt', 'w') as f:
        f.write(f"accuracy: {acc:.4f}\n")
    return cm, acc

# fluxo principal
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    X, Y = load_data()
    # split: últimos TEST_SIZE para teste
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, TEST_SIZE)
    # grid search com cross-validation no conjunto de treino
    best_cfg = grid_search(X_train, Y_train)
    # treino final com melhor configuração
    model = MLP(X_train.shape[1], best_cfg['hidden'], Y.shape[1],
                best_cfg['lr'], SEED, best_cfg['patience'], best_cfg['epochs'])
    # salva e depois treina
    prefix = os.path.join(OUTDIR, 'dataset_best')
    np.save(prefix + '_weights_initial_W1.npy', model.W1)
    np.save(prefix + '_weights_initial_W2.npy', model.W2)
    model.train(X_train, Y_train)
    np.save(prefix + '_weights_final_W1.npy', model.W1)
    np.save(prefix + '_weights_final_W2.npy', model.W2)
    # avaliação no teste
    cm, acc = evaluate_and_save(model, X_test, Y_test, prefix)
    logging.info(f"Test accuracy: {acc:.4f}")