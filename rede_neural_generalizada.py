# =============================================================================
# rede_neural_generalizada.py
#
# Trabalho de Rede Neural — Generalização do exemplo4.py
#
# Autor(es): Maurício Rodrigues da Silva
# Data: 25/09/2026
# E-mail: 
#
# OBJETIVO
# --------
# Reescrever a rede neural do arquivo exemplo4.py de forma GENÉRICA e
# VETORIZADA, permitindo alterar facilmente:
#   - o dataset
#   - o número de neurônios da camada oculta
#   - as funções de ativação
#   - o otimizador (GD com momento)
#   - a estratégia de treino (mini-batch)
#
# Além disso, o código faz a VERIFICAÇÃO NUMÉRICA do gradiente
# analítico (via diferenças finitas centradas), garantindo que a
# implementação do backpropagation está correta.
#
# DIFERENÇAS EM RELAÇÃO AO exemplo4.py ORIGINAL
# ---------------------------------------------
#   1. Dataset: make_moons -> make_circles (mais difícil)
#   2. Camada oculta: 2 neurônios -> 8 neurônios
#   3. Ativação oculta: sigmoid -> tanh
#   4. Forward/backward: código hard-coded -> notação vetorial (matrizes)
#   5. Otimizador: GD puro -> GD com momento
#   6. Treino: batch completo -> mini-batch
#   7. Extras: verificação numérica de gradiente, curva de loss e
#      fronteira de decisão
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# =============================================================================
# 1. DATASET
# =============================================================================
# Utilizamos o dataset make_circles, que consiste em dois círculos
# concêntricos de classes diferentes. É um problema NÃO linearmente
# separável, exigindo pelo menos uma camada oculta para ser resolvido.
#
# Também normalizamos os dados (média 0, desvio 1), o que ajuda o
# treinamento a convergir mais rápido e evita saturação das ativações.
# =============================================================================
N = 200
X, Y = datasets.make_circles(n_samples=N, noise=0.1, factor=0.5)

# Normalização (z-score): x' = (x - média) / desvio padrão
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Y passa a ter formato (N, 1) para facilitar o cálculo do erro como vetor
Y = Y.reshape(-1, 1)

# Plot inicial do dataset (apenas para o relatório)
color = ['blue' if k == 0 else 'red' for k in Y.ravel()]
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=color, edgecolor='k')
plt.title('Dataset make_circles (normalizado)')
plt.savefig('circles.svg')
plt.close()


# =============================================================================
# 2. FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
# =============================================================================
# Cada ativação é definida por duas funções:
#   - a própria ativação: f(v)
#   - sua derivada: f'(y), calculada a partir da SAÍDA y (não da entrada v),
#     o que é mais eficiente e numericamente estável.
#
# Observação: para ReLU, a derivada depende de v (entrada), não de y.
# Por isso incluímos a versão alternativa comentada.
# =============================================================================
def sigmoid(v):
    """Ativação sigmoide: 1 / (1 + exp(-v))"""
    return 1.0 / (1.0 + np.exp(-v))


def sigmoid_prime(y):
    """Derivada da sigmoide em função de sua saída: y * (1 - y)"""
    return y * (1.0 - y)


def tanh(v):
    """Ativação tangente hiperbólica"""
    return np.tanh(v)


def tanh_prime(y):
    """Derivada da tanh em função de sua saída: 1 - y^2"""
    return 1.0 - y ** 2


def relu(v):
    """Ativação ReLU: max(0, v)"""
    return np.maximum(0.0, v)


def relu_prime(v):
    """Derivada da ReLU em função da ENTRADA v (1 se v>0, senão 0)"""
    return (v > 0).astype(float)


# -------- Escolha das ativações usadas nesta rede --------
# Camada oculta: tanh (suave, centrada em zero, evita saturação)
# Camada de saída: sigmoid (produz saída entre 0 e 1, ideal p/ binário)
f, f_prime = tanh, tanh_prime
g, g_prime = sigmoid, sigmoid_prime


# =============================================================================
# 3. FORWARD PASS GENÉRICO
# =============================================================================
# Recebe uma amostra x (vetor de entrada) e os parâmetros da rede.
# Retorna o cache com valores intermediários necessários ao backward.
#
# Arquitetura:
#   x (d,) -> v0 = W0 @ x + b0 (n_hidden,) -> y0 = f(v0)
#          -> v1 = W1 @ y0 + b1 (1,)        -> y1 = g(v1)
# =============================================================================
def forward(x, params):
    """Executa o forward pass para uma amostra x.

    Parâmetros
    ----------
    x : np.ndarray, shape (d,)
        Vetor de entrada.
    params : tuple (W0, b0, W1, b1)
        W0 : (n_hidden, d)  pesos da camada oculta
        b0 : (n_hidden,)    bias da camada oculta
        W1 : (1, n_hidden)  pesos da camada de saída
        b1 : (1,)           bias da camada de saída

    Retorna
    -------
    v0, y0, v1, y1 : valores intermediários (cache)
    """
    W0, b0, W1, b1 = params

    v0 = W0 @ x + b0          # (n_hidden,) - soma ponderada da oculta
    y0 = f(v0)                # (n_hidden,) - ativação da oculta
    v1 = W1 @ y0 + b1         # (1,)        - soma ponderada da saída
    y1 = g(v1)                # (1,)        - ativação final (previsão)

    return (v0, y0, v1, y1)


# =============================================================================
# 4. LOSS + BACKWARD PASS GENÉRICO
# =============================================================================
# Calcula a perda MSE para uma amostra e os gradientes de TODOS os
# parâmetros pela regra da cadeia.
#
# Derivação (notação):
#   L  = 1/2 * (y1 - d)^2
#   dL/dy1 = (y1 - d)
#   dL/dv1 = dL/dy1 * g'(y1)              (regra da cadeia na ativação)
#   dL/dW1 = dL/dv1 * y0^T                (produto externo)
#   dL/db1 = dL/dv1
#   dL/dy0 = W1^T @ dL/dv1                (propaga o erro para trás)
#   dL/dv0 = dL/dy0 * f'(y0)              (Hadamard)
#   dL/dW0 = dL/dv0 * x^T                 (produto externo)
#   dL/db0 = dL/dv0
# =============================================================================
def loss_and_grads(x, d, params):
    """Calcula loss + gradientes para uma amostra.

    Parâmetros
    ----------
    x : (d,)       entrada
    d : (1,)       rótulo desejado
    params : (W0, b0, W1, b1)

    Retorna
    -------
    L : float
    grads : tuple (grad_W0, grad_b0, grad_W1, grad_b1)
    """
    W0, b0, W1, b1 = params

    # ---------- FORWARD ----------
    v0, y0, v1, y1 = forward(x, params)

    # Erro e perda MSE
    e = y1 - d                                  # (1,)
    L = 0.5 * float((e ** 2).item())

    # ---------- BACKWARD ----------
    # Camada de saída
    grad_v1 = e * g_prime(y1)                   # (1,)
    grad_W1 = np.outer(grad_v1, y0)             # (1, n_hidden)
    grad_b1 = grad_v1                           # (1,)

    # Camada oculta
    grad_y0 = W1.T @ grad_v1                    # (n_hidden,)
    grad_v0 = grad_y0 * f_prime(y0)             # (n_hidden,)  Hadamard
    grad_W0 = np.outer(grad_v0, x)              # (n_hidden, d)
    grad_b0 = grad_v0                           # (n_hidden,)

    return L, (grad_W0, grad_b0, grad_W1, grad_b1)


# =============================================================================
# 5. VERIFICAÇÃO NUMÉRICA DO GRADIENTE
# =============================================================================
# Para garantir que o backward está correto, comparamos o gradiente
# analítico com a aproximação por diferenças finitas centradas:
#
#   dL/dtheta ≈ (L(theta + eps) - L(theta - eps)) / (2 * eps)
#
# Se o erro relativo for < 1e-6, a implementação está correta.
# =============================================================================
def numerical_grad(x, d, params, eps=1e-5):
    """Aproxima o gradiente por diferenças finitas centradas."""
    grads_num = []

    # Itera sobre cada matriz/vetor de parâmetros
    for p in params:
        g = np.zeros_like(p)

        # Itera sobre cada elemento escalar do parâmetro
        it = np.nditer(p, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]

            # L(theta + eps)
            p[idx] = orig + eps
            L_plus, _ = loss_and_grads(x, d, params)

            # L(theta - eps)
            p[idx] = orig - eps
            L_minus, _ = loss_and_grads(x, d, params)

            # Restaura valor original
            p[idx] = orig

            # Derivada numérica
            g[idx] = (L_plus - L_minus) / (2 * eps)
            it.iternext()

        grads_num.append(g)

    return grads_num


def check_gradients():
    """Compara gradiente analítico e numérico para uma amostra."""
    print("=" * 60)
    print("VERIFICAÇÃO NUMÉRICA DO GRADIENTE")
    print("=" * 60)

    # Inicialização reprodutível
    np.random.seed(0)
    d_in = X.shape[1]
    n_hidden = 8

    W0 = np.random.randn(n_hidden, d_in) * 0.1
    b0 = np.zeros(n_hidden)
    W1 = np.random.randn(1, n_hidden) * 0.1
    b1 = np.zeros(1)
    params = [W0, b0, W1, b1]

    # Amostra de teste
    x = X[0]
    d = Y[0]

    # Gradiente analítico
    _, grads_an = loss_and_grads(x, d, params)

    # Gradiente numérico
    grads_num = numerical_grad(x, d, params)

    # Compara cada parâmetro
    nomes = ['W0', 'b0', 'W1', 'b1']
    for nome, ga, gn in zip(nomes, grads_an, grads_num):
        denom = np.maximum(np.abs(ga) + np.abs(gn), 1e-12)
        rel_err = np.max(np.abs(ga - gn) / denom)
        print(f"  {nome:3s} | erro relativo máx: {rel_err:.2e}")

    print("=" * 60)


# Executa a verificação
check_gradients()


# =============================================================================
# 6. TREINAMENTO
# =============================================================================
# Estratégia:
#   - Mini-batch de tamanho batch_size
#   - Otimizador: GD com momento (momentum)
#       v <- momentum * v - lr * grad
#       theta <- theta + v
#   - Embaralhamento das amostras a cada época
# =============================================================================
def train(epochs=200, batch_size=16, lr=0.1, momentum=0.9):
    """Treina a rede e retorna os parâmetros finais + histórico de loss."""

    np.random.seed(42)
    d_in = X.shape[1]
    n_hidden = 8

    # ---------- Inicialização ----------
    # randn * 0.1 -> pequenos valores, evita saturação de tanh/sigmoid
    W0 = np.random.randn(n_hidden, d_in) * 0.1
    b0 = np.zeros(n_hidden)
    W1 = np.random.randn(1, n_hidden) * 0.1
    b1 = np.zeros(1)
    params = [W0, b0, W1, b1]

    # Velocidades para o momento (mesmo shape dos parâmetros)
    vW0 = np.zeros_like(W0)
    vb0 = np.zeros_like(b0)
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vels = [vW0, vb0, vW1, vb1]

    losses = []         # histórico da loss média por época
    n = X.shape[0]      # número de amostras

    # ---------- Loop de épocas ----------
    for ep in range(epochs):
        # Embaralha amostras a cada época
        idx = np.random.permutation(n)
        X_sh, Y_sh = X[idx], Y[idx]
        ep_loss = 0.0

        # ---------- Loop de mini-batches ----------
        for i in range(0, n, batch_size):
            xb = X_sh[i:i + batch_size]
            yb = Y_sh[i:i + batch_size]

            # Acumuladores de gradiente do mini-batch
            gW0 = np.zeros_like(W0)
            gb0 = np.zeros_like(b0)
            gW1 = np.zeros_like(W1)
            gb1 = np.zeros_like(b1)
            gs = [gW0, gb0, gW1, gb1]
            b_loss = 0.0

            # Soma os gradientes de cada amostra do mini-batch
            for xi, di in zip(xb, yb):
                L, grads = loss_and_grads(xi, di, params)
                b_loss += L
                for g_acc, g_i in zip(gs, grads):
                    g_acc += g_i

            # Média (gradiente do mini-batch)
            for g_acc in gs:
                g_acc /= len(xb)

            # ---------- Atualização com momento ----------
            for p, g, v in zip(params, gs, vels):
                v[:] = momentum * v - lr * g
                p += v

            ep_loss += b_loss

        # Loss média da época
        losses.append(ep_loss / n)

        # Log a cada 20 épocas
        if ep % 20 == 0:
            print(f"Época {ep:4d} | loss média: {losses[-1]:.4f}")

    return params, losses


# Executa o treinamento
params, losses = train()


# =============================================================================
# 7. AVALIAÇÃO (ACURÁCIA)
# =============================================================================
def predict(x, params):
    """Retorna 0 ou 1 com base na saída da rede."""
    _, _, _, y1 = forward(x, params)
    return 1 if y1[0] > 0.5 else 0


acertos = sum(predict(X[i], params) == Y[i, 0] for i in range(N))
print(f"\nAcurácia final: {acertos}/{N} = {acertos/N:.2%}")


# =============================================================================
# 8. CURVA DE LOSS
# =============================================================================
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel('Época')
plt.ylabel('Loss (MSE média)')
plt.title('Curva de aprendizado')
plt.grid(True)
plt.savefig('curva_loss.svg')
plt.close()


# =============================================================================
# 9. FRONTEIRA DE DECISÃO
# =============================================================================
# Avalia a rede em uma grade densa de pontos no plano (x1, x2) e
# colore cada região conforme a classe prevista. Isso permite
# visualizar COMO a rede separa as classes.
# =============================================================================
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 200),
    np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 200),
)

# Avalia a rede ponto a ponto na grade
Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z[i, j] = predict(np.array([xx[i, j], yy[i, j]]), params)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=color, edgecolor='k')
plt.title(f'Fronteira de decisão — acc = {acertos/N:.2%}')
plt.savefig('fronteira_decisao.svg')
plt.close()

print("\nFiguras salvas: circles.svg, curva_loss.svg, fronteira_decisao.svg")