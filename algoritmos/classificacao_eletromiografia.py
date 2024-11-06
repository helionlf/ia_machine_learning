import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
data = np.loadtxt("venv/algoritmos/EMGsDataset.csv", delimiter=',')
x = data[:2, :].T
N, p = x.shape

# Organizar os dados X e Y
X = np.concatenate((np.ones((N, 1)), x), axis=1)
Y = data[2, :].astype(int)  # Classes reais

# Funções utilitárias
def calcular_media(X):
    return np.mean(X, axis=0)

def calcular_covariancia(X, regularizacao=1e-6):
    cov = np.cov(X, rowvar=False)
    return cov + regularizacao * np.eye(cov.shape[0])

# MQO tradicional
def mqo(X, Y):
    X_T = X.T
    W = np.linalg.inv(X_T @ X) @ X_T @ np.eye(5)[Y - 1]
    return W

def predict_mqo(X, W):
    return np.argmax(X @ W, axis=1) + 1  # +1 para mapear as classes

# Classificador Gaussiano Tradicional
def classificador_gaussiano_tradicional(X, y):
    classes = np.unique(y)
    medias = {}
    covariancias = {}
    for c in classes:
        X_c = X[y == c]
        medias[c] = calcular_media(X_c)
        covariancias[c] = calcular_covariancia(X_c)
    return medias, covariancias

def predict_gaussiano_tradicional(X, medias, covariancias):
    probabilidades = []
    for c, media in medias.items():
        cov = covariancias[c]
        inv_cov = np.linalg.inv(cov)
        prob = -0.5 * np.sum((X - media) @ inv_cov * (X - media), axis=1)
        prob -= 0.5 * np.log(np.linalg.det(cov) + 1e-6)
        probabilidades.append(prob)
    return np.argmax(np.vstack(probabilidades).T, axis=1) + 1

# Classificador Gaussiano com Covariâncias Iguais
def classificador_gaussiano_cov_iguais(X, y):
    classes = np.unique(y)
    medias = {}
    covariancia_igual = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        X_c = X[y == c]
        medias[c] = calcular_media(X_c)
        covariancia_igual += calcular_covariancia(X_c) * len(X_c)
    covariancia_igual /= len(X)
    return medias, covariancia_igual

def predict_gaussiano_cov_iguais(X, medias, cov_igual):
    inv_cov = np.linalg.inv(cov_igual + 1e-6 * np.eye(cov_igual.shape[0]))  # Regularização
    probabilidades = []
    for media in medias.values():
        prob = -0.5 * np.sum((X - media) @ inv_cov * (X - media), axis=1)
        prob -= 0.5 * np.log(np.linalg.det(cov_igual) + 1e-6)
        probabilidades.append(prob)
    return np.argmax(np.vstack(probabilidades).T, axis=1) + 1

# Classificador Gaussiano com Matriz Agregada
def classificador_gaussiano_matriz_agregada(X):
    media_geral = calcular_media(X)
    covariancia_agregada = calcular_covariancia(X)
    return media_geral, covariancia_agregada

def predict_gaussiano_matriz_agregada(X, media_geral, cov_agregado):
    inv_cov = np.linalg.inv(cov_agregado + 1e-6 * np.eye(cov_agregado.shape[0]))  # Regularização
    prob = -0.5 * np.sum((X - media_geral) @ inv_cov * (X - media_geral), axis=1)
    return np.ones(len(X)) if np.allclose(prob, prob[0]) else np.argmax(prob) + 1

# Classificador Gaussiano Regularizado
def classificador_gaussiano_regularizado(X, y, lamb):
    classes = np.unique(y)
    medias = {}
    covariancia_igual = np.zeros((X.shape[1], X.shape[1]))
    for c in classes:
        X_c = X[y == c]
        medias[c] = calcular_media(X_c)
        covariancia_igual += calcular_covariancia(X_c) * len(X_c)
    covariancia_igual /= len(X)
    covariancia_regularizada = lamb * np.eye(X.shape[1]) + (1 - lamb) * covariancia_igual
    return medias, covariancia_regularizada

def predict_gaussiano_regularizado(X, medias, cov_regularizada):
    inv_cov = np.linalg.inv(cov_regularizada + 1e-6 * np.eye(cov_regularizada.shape[0]))  # Regularização
    probabilidades = []
    for media in medias.values():
        prob = -0.5 * np.sum((X - media) @ inv_cov * (X - media), axis=1)
        prob -= 0.5 * np.log(np.linalg.det(cov_regularizada) + 1e-6)
        probabilidades.append(prob)
    return np.argmax(np.vstack(probabilidades).T, axis=1) + 1

# Função de acurácia
def calcular_acuracia(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)

# Função de simulação Monte Carlo
R = 500
acuracias_mqo = []
acuracias_gaussiano = []
acuracias_gaussiano_cov_iguais = []
acuracias_gaussiano_matriz_agregada = []
acuracias_gaussiano_regularizado = []

for _ in range(R):
    indices = np.random.permutation(N)
    X_embaralhado = X[indices]
    y_embaralhado = Y[indices]
    split_index = int(0.8 * N)
    X_treino, X_teste = X_embaralhado[:split_index], X_embaralhado[split_index:]
    y_treino, y_teste = y_embaralhado[:split_index], y_embaralhado[split_index:]

    # MQO
    W_mqo = mqo(X_treino, y_treino)
    Y_pred_mqo = predict_mqo(X_teste, W_mqo)
    acuracias_mqo.append(calcular_acuracia(y_teste, Y_pred_mqo))

    # Gaussiano Tradicional
    medias_gauss, covariancias_gauss = classificador_gaussiano_tradicional(X_treino, y_treino)
    Y_pred_gauss = predict_gaussiano_tradicional(X_teste, medias_gauss, covariancias_gauss)
    acuracias_gaussiano.append(calcular_acuracia(y_teste, Y_pred_gauss))

    # Gaussiano com Covariâncias Iguais
    medias_gauss_iguais, cov_igual = classificador_gaussiano_cov_iguais(X_treino, y_treino)
    Y_pred_gauss_iguais = predict_gaussiano_cov_iguais(X_teste, medias_gauss_iguais, cov_igual)
    acuracias_gaussiano_cov_iguais.append(calcular_acuracia(y_teste, Y_pred_gauss_iguais))

    # Gaussiano com Matriz Agregada
    media_geral, cov_agregado = classificador_gaussiano_matriz_agregada(X_treino)
    Y_pred_gauss_agregado = predict_gaussiano_matriz_agregada(X_teste, media_geral, cov_agregado)
    acuracias_gaussiano_matriz_agregada.append(calcular_acuracia(y_teste, Y_pred_gauss_agregado))

    # Gaussiano Regularizado
    lamb = 0.5  # Testando com lambda fixo
    medias_gauss_reg, cov_regularizada = classificador_gaussiano_regularizado(X_treino, y_treino, lamb)
    Y_pred_gauss_reg = predict_gaussiano_regularizado(X_teste, medias_gauss_reg, cov_regularizada)
    acuracias_gaussiano_regularizado.append


# Estatísticas finais
def calcular_estatisticas(acuracias):
    if len(acuracias) == 0:
        return {
            "média": np.nan,
            "desvio padrão": np.nan,
            "valor máximo": np.nan,
            "valor mínimo": np.nan
        }
    return {
        "média": np.mean(acuracias),
        "desvio padrão": np.std(acuracias),
        "valor máximo": np.max(acuracias),
        "valor mínimo": np.min(acuracias)
    }

print("Estatísticas MQO:", calcular_estatisticas(acuracias_mqo))
print("Estatísticas Gaussiano Tradicional:", calcular_estatisticas(acuracias_gaussiano))
print("Estatísticas Gaussiano com Covariâncias Iguais:", calcular_estatisticas(acuracias_gaussiano_cov_iguais))
print("Estatísticas Gaussiano com Matriz Agregada:", calcular_estatisticas(acuracias_gaussiano_matriz_agregada))
print("Estatísticas Gaussiano Regularizado:", calcular_estatisticas(acuracias_gaussiano_regularizado))
