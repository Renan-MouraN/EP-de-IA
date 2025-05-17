import numpy as np
import os
import csv
import logging
from itertools import product

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======= Parâmetros ========
X_NPY_PATH    = 'X.npy'           # caminho do arquivo X.npy (shape: N,10,12,1)
Y_NPY_PATH    = 'Y_classe.npy'    # caminho do arquivo Y_classe.npy (shape: N,26)
OUTDIR        = 'output'          # diretório de saída
TEST_RATIO    = 0.2               # proporção de dados para teste
SEED          = 42                # semente aleatória
CV_FOLDS      = 5                 # número de folds para cross-validation

# Parâmetros para grid search
GRID_HIDDEN   = [25, 50, 75, 100]       # números de neurônios na camada oculta
GRID_LR       = [0.001, 0.01, 0.05, 0.1]# diferentes learning rates
GRID_EPOCHS   = [50, 100, 150]         # diferentes números de épocas
GRID_PATIENCE = [5, 10, 15]           # diferentes paciências para early stopping
# ============================

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 learning_rate=0.01, seed=None, patience=None, max_epochs=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs
        # inicializa pesos (inclui bias)
        self.W1 = np.random.randn(n_hidden, n_inputs + 1) * 0.1
        self.W2 = np.random.randn(n_outputs, n_hidden + 1) * 0.1

    @staticmethod
    def _activate(x): return np.tanh(x)
    @staticmethod
    def _activate_derivative(y): return 1.0 - y**2
    @staticmethod
    def _activateOutput(x): 
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def feedforward(self, X):
        xb = np.append(X, 1.0)
        self.z = self._activate(self.W1.dot(xb))
        zb = np.append(self.z, 1.0)
        #Usa SoftMax Para Camada de Saída
        self.y = self._activateOutput(self.W2.dot(zb))
        return self.y

    # Mudei para isso mas não sei se é válido
    # É algo relacionado a função soft max
    def backprop_step(self, X, T):
        Y = self.feedforward(X)
        # gradiente da saída para softmax + cross-entropy simplifica para (Y - T)
        delta_out = Y - T
        delta_hidden = self._activate_derivative(self.z) * self.W2[:, :-1].T.dot(delta_out)
        self.W2 -= self.lr * np.outer(delta_out, np.append(self.z, 1.0))  # note o '-=' para grad descent
        self.W1 -= self.lr * np.outer(delta_hidden, np.append(X, 1.0))
        return -np.sum(T * np.log(Y + 1e-15))  # cross-entropy loss


    # def backprop_step(self, X, T):
    #     Y = self.feedforward(X)
    #     delta_out = (T - Y) * self._activate_derivative(Y)
    #     delta_hidden = self._activate_derivative(self.z) * self.W2[:, :-1].T.dot(delta_out)
    #     self.W2 += self.lr * np.outer(delta_out, np.append(self.z, 1.0))
    #     self.W1 += self.lr * np.outer(delta_hidden, np.append(X, 1.0))
    #     return 0.5 * np.sum((T - Y)**2)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        best_val = float('inf')
        epochs_no_improve = 0
        for epoch in range(1, self.max_epochs + 1):
            train_error = sum(self.backprop_step(x, t) for x, t in zip(X_train, y_train))
            if X_val is not None:
                val_error = sum(0.5 * np.sum((t - self.feedforward(x))**2)
                                for x, t in zip(X_val, y_val))
                if self.patience and val_error < best_val:
                    best_val, epochs_no_improve = val_error, 0
                    bestW1, bestW2 = self.W1.copy(), self.W2.copy()
                else:
                    epochs_no_improve += 1
                if self.patience and epochs_no_improve >= self.patience:
                    self.W1, self.W2 = bestW1, bestW2
                    break
        return train_error, (val_error if X_val is not None else None)

    def predict(self, X):
        out = self.feedforward(X)
        idx = np.argmax(out)
        vec = -np.ones_like(out)
        vec[idx] = 1
        return vec

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

# === Carregamento e pré-processamento ===
def load_data():
    X = np.load(X_NPY_PATH)  # (N,10,12,1)
    Y = np.load(Y_NPY_PATH)  # (N,26)
    N = X.shape[0]
    X = X.reshape(N, -1)      # flatten
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalize
    # Adicionando uma coluna de 1 no início
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    Y = np.where(Y > 0, 1, -1)
    return X[:-130, :], Y[:-130], X[-130:, :], Y[-130:]

# === Split treino/teste ===
def train_test_split(X, Y, test_ratio, seed):
    np.random.seed(seed)
    N = X.shape[0]
    idx = np.random.permutation(N)
    t_end = int(N * test_ratio)
    return X[idx[t_end:]], Y[idx[t_end:]], X[idx[:t_end]], Y[idx[:t_end]]

# === Avaliação e salvamento ===
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

# === Cross-validation manual para grid search ===
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
        _, _ = model.train(X_tr, Y_tr, X_val, Y_val)
        acc = np.mean((model.predict_batch(X_val) == Y_val).all(axis=1))
        scores.append(acc)
        current = stop
    return np.mean(scores)

# === Grid search com cross-validation ===
def grid_search(X, Y):
    results = []
    for hidden, lr, epochs, pat in product(GRID_HIDDEN, GRID_LR, GRID_EPOCHS, GRID_PATIENCE):
        params = {'hidden':hidden, 'lr':lr, 'epochs':epochs, 'patience':pat}
        cv_score = cross_validate(X, Y, params)
        results.append({**params, 'cv_acc':cv_score})
    # salva resultados
    os.makedirs(OUTDIR, exist_ok=True)
    keys = results[0].keys()
    with open(os.path.join(OUTDIR, 'grid_search.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    best = max(results, key=lambda x: x['cv_acc'])
    logging.info(f"Best CV config: {best}")
    return best

# === Fluxo principal ===
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    # Carrega dados
    X_train, Y_train, X_test , Y_test = load_data()
    # Split treino/teste
    # Ela pediu para usar os ultimos 130 como teste ent tirei isso 
    # X_train, Y_train, X_test, Y_test = train_test_split(X, Y, TEST_RATIO, SEED)
    # Grid search com cross-validation no conjunto de treino
    best_cfg = grid_search(X_train, Y_train)
    # Treino final com melhor configuração em todo o treino
    prefix = os.path.join(OUTDIR, 'dataset_best')

    # Inicializa modelo e salva pesos iniciais
    model = MLP(
        X_train.shape[1], best_cfg['hidden'], Y_train.shape[1],
        best_cfg['lr'], SEED, best_cfg['patience'], best_cfg['epochs']
    )
    os.makedirs(OUTDIR, exist_ok=True)
    np.save(prefix + '_weights_initial_W1.npy', model.W1)
    np.save(prefix + '_weights_initial_W2.npy', model.W2)

    # Treina modelo
    model.train(X_train, Y_train)

    # Salva pesos finais
    np.save(prefix + '_weights_final_W1.npy', model.W1)
    np.save(prefix + '_weights_final_W2.npy', model.W2)

    # Avaliação teste
    cm, acc = evaluate_and_save(model, X_test, Y_test, prefix)
    logging.info(f"Test accuracy: {acc:.4f}")