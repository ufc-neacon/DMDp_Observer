import numpy as np
import matplotlib.pyplot as plt

# Funções do modelo do tanque e Jacobiana
def tank_model_discrete(xk, uk):
    """ Modelo discreto do sistema (tanque) """
    r = 0.635
    R = 0.73
    H = 1.38
    q_nominal = 14 / 3600
    ki = q_nominal / (2 * np.pi * 60)
    ko = 0.6 * ki * 2 * np.pi * 60 / (np.sqrt(0.60 * H))

    xk = max(xk, 1e-6)  # Evita sqrt(negativo)
    xk1 = xk + (ki * uk - ko * np.sqrt(xk)) / (
        np.pi * r**2 + 2 * np.pi * r * xk * (R - r) / H
        + np.pi * xk**2 * (R - r)**2 / H**2
    )
    return max(min(xk1, H), 0)  # Garante que x está dentro dos limites

def jacobian_tank(xk, uk):
    """ Cálculo numérico da Jacobiana """
    epsilon = 1e-5
    f_x_plus = tank_model_discrete(xk + epsilon, uk)
    f_x_minus = tank_model_discrete(xk - epsilon, uk)
    return (f_x_plus - f_x_minus) / (2 * epsilon)

class EKF:
    def __init__(self, f_model, jacobian_f, P0=0.1):
        self.P = np.array([P0], dtype=np.float64)  # Covariância inicial
        self.f_model = f_model  # Função de predição do modelo
        self.jacobian_f = jacobian_f  # Função Jacobiana
    
    def step(self, x_prev, u, y_measured, Q, R):
        F = self.jacobian_f(x_prev, u)  # Jacobiana
        x_pred = self.f_model(x_prev, u)  # Estado previsto
        P_pred = F * self.P * F + Q  # Atualização da covariância

        H = 1  # Matriz de observação (y = x)
        S = H * P_pred * H + R  # Inovação
        K = P_pred * H / S  # Ganho de Kalman
        x_est = x_pred + K * (y_measured - H * x_pred)  # Atualiza a estimativa
        P_est = (1 - K * H) * P_pred  # Atualiza a covariância
        
        self.P = P_est  # Atualiza a covariância para o próximo passo
        return x_est  # Retorna o estado estimado
    
T_max = 20000
dt = 1
Ta = np.arange(0, T_max, dt)
cu = 1  

x_real = 0
x_est = 0.105
xT_real = []
xT_est = []
uT = []
ruido = []
erro_medio_T = []

ekf = EKF(f_model=tank_model_discrete, jacobian_f=jacobian_tank)

for i in Ta:
    u = cu * np.pi * 60 * (1 - np.cos(2 * np.pi * i / 2500))
    x_real = tank_model_discrete(x_real, u)
    eta = 0*1e-2 * (np.random.rand() - 0.5)
    y_measured = x_real + eta  # Ruído deve se propagar
    
    ruido.append(eta)
    xT_real.append(x_real)
    uT.append(u)
    
    R_est = 0.1325 #np.var(np.array(ruido)) if len(ruido) > 1 else 0.1
    Q_est = 1 #np.var(np.diff(xT_real)) if len(xT_real) > 1 else 0.01
    
    x_est = ekf.step(x_est, u, x_real, Q_est, R_est) 
    xT_est.append(x_est)
    
xT_real = np.array(xT_real) 
xT_est = np.array(xT_est)[:, 0]
e_2k = np.abs(xT_real - xT_est)
print(np.mean(e_2k))

plt.figure()
plt.plot(Ta[0:11], e_2k[0:11], label="Erro", color="green")
plt.xlabel("Tempo")
plt.ylabel("Erro")
plt.title("Evolução do Erro")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(Ta, xT_real, label="Estado Real (h)", color="blue")
plt.plot(Ta, xT_est, label="Estado Estimado (EKF)", linestyle="dashed", color="red")
plt.xlabel("Tempo")
plt.ylabel("Altura do fluido (h)")
plt.legend()
plt.title("Comparação entre o estado real e o estimado pelo EKF")
plt.grid()
plt.show()
