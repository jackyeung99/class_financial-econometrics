# from typing import tuple, int
from sklearn.linear_model import LinearRegression
import numpy as np



'''
1. Intitialize p and q parameters
    Fit OLS estimator of an AR model
    Fit MA coefficients to the residuals
2. Convert ARMA coefficients into state-space model
3. compute log likelihood 
4. maximize state space parameters w.r.t to log likelihood

'''
class ARIMA_model():

    def __init__(self, time_series: np.array,  order: tuple[int, int, int]) :
        self.p, self.i, self.q = order
        self.y = self.demean(time_series)


    def demean(self, l, time_series): 
        for _ in l:
            time_series = time_series - np.mean(time_series)

        return time_series


    def ols_fit(self, y, lag):
        p = lag 
        Y = y[p:]
        #lag matrix
        X = np.column_stack([y[p-i-1: -i-1] for i in range(p)])
        
        # OLS estimation
        XtX = X.T @ X                          
        XtY = X.T @ Y                          
        beta_hat = np.linalg.solve(XtX, XtY)  

        return beta_hat



    def init_parameters(self):
        # call AR 
        alpha_hat = self.ols_fit(self.y, self.p)
        # can estimate MA parameetrs using Hannan-Rissanen, keeping simple since we will update using Kalman Filter
        beta_hat = np.array([.1 for x in range(self.q)])
        return alpha_hat, beta_hat
    

    def build_state_space_model(alpha, beta, sigma2):

        pass



    def optimize_parameters():
        pass


    def fit(time_series):

        model = None

        a_z = sum()
        b_z = sum()

        

        return model

    def compute_likelihood():
        pass

    def compute_aic():
        pass

    def compute_bic():
        pass


    def check_roots(coefs):
        # coefs: AR poly phi(z)=1 - phi1 z - ... - phip z^p  (same for MA)
        if len(coefs) == 0:
            return True
        poly = np.r_[1.0, -coefs]
        r = np.roots(poly)
        return np.all(np.abs(r) > 1.0 + 1e-6)