import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:/Users/helio/OneDrive/Documentos/meus_projetos/ia_achine_learning/venv/algoritmos/aerogerador.dat", delimiter='\t')
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
N, p = x.shape

X = np.concatenate((np.ones((N, 1)), x), axis=1)

lambdas = [0, 0.25, 0.5, 0.75, 1]
W_tikhonov = []

plt.scatter(x, y, color='blue', edgecolor='k', label='Pontos')

for i in lambdas:
    W_reg = np.linalg.inv(X.T @ X + i * np.eye(p+1)) @ X.T @ y
    W_tikhonov.append(W_reg)
    y_hat_reg = X @ W_reg
    plt.plot(x, y_hat_reg, label=f'MQO Regularizado (lambda={i})')


R = 500 
rss_values_mqo = np.empty((0, len(lambdas)))
rss_values_media = []

for r in range(R):
    indices = np.random.permutation(N)
    X_embaralhado = X[indices]
    y_embaralhado = y[indices]

    split_index = int(0.8 * N)  # 80% para treino
    X_treino, X_teste = X_embaralhado[:split_index], X_embaralhado[split_index:]
    y_treino, y_teste = y_embaralhado[:split_index], y_embaralhado[split_index:]

    # RSS para cada lambda
    rss_lambdas = np.zeros((1, len(lambdas)))
    for i, lambda_ in enumerate(lambdas):
        W_regularizado = np.linalg.inv(X_treino.T @ X_treino + lambda_ * np.eye(X_treino.shape[1])) @ X_treino.T @ y_treino
        y_predito = X_teste @ W_regularizado
        rss_regularizado = np.sum((y_predito - y_teste) ** 2)
        rss_lambdas[0, i] = rss_regularizado
    
    rss_values_mqo = np.concatenate((rss_values_mqo, rss_lambdas))

    y_media = np.mean(y_treino)
    y_predito_media = np.full_like(y_teste, y_media)
    rss_media = np.sum((y_predito_media - y_teste) ** 2)
    rss_values_media.append(rss_media)

# Estatísticas
rss_mean = np.mean(rss_values_media, axis=0)
rss_std = np.std(rss_values_media, axis=0)
rss_min = np.min(rss_values_media, axis=0)
rss_max = np.max(rss_values_media, axis=0)

# Estatísticas valores observáveis
print("Média dos valores observáveis :", rss_mean)
print("Desvio padrão dos valores observáveis:", rss_std)
print("Valor mínimo dos valores observáveis", rss_min)
print("Valor máximo dos valores observáveis", rss_max)

rss_mean = np.mean(rss_values_mqo, axis=0)
rss_std = np.std(rss_values_mqo, axis=0)
rss_min = np.min(rss_values_mqo, axis=0)
rss_max = np.max(rss_values_mqo, axis=0)

# Estatísticas de RSS
print("Média do RSS para cada lambda:", rss_mean)
print("Desvio padrão do RSS para cada lambda:", rss_std)
print("Valor mínimo do RSS para cada lambda:", rss_min)
print("Valor máximo do RSS para cada lambda:", rss_max)

# Configurações finais do gráfico
plt.xlabel("Medida de Velocidade do Vento")
plt.ylabel("Potência Gerada pelo Aerogerador")
plt.title("Modelos de Regressão e Estatísticas de RSS")
plt.legend()
plt.show()
