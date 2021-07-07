import numpy as np

class LD():
    def __init__(self):
        pass

    def update(self, x0, dlogprob, n_iter = 1000, stepsize = 1e-3, alpha = 0.9, adagrad=True):
        init = np.copy(x0)
        theta_list = [init]

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            logpgrad = dlogprob(init)

            direct = 1 / 2 * logpgrad + np.random.normal(0, 1, [init.shape[0], init.shape[1]]) / np.sqrt(stepsize)
            
            if adagrad:
                # adagrad 
                if iter == 0:
                    historical_grad = historical_grad + direct ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (direct ** 2)
                adj_grad = np.divide(direct, fudge_factor+np.sqrt(historical_grad))
                init = init + stepsize * adj_grad
            else:
                init = init + stepsize * direct

            theta_list.append(init)

        return np.array(theta_list)