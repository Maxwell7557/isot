import pandas as pd
import pystan
#from utils.psis import psisloo

answers_A = pd.read_csv('/home/katya/Downloads/output/data/A.csv')
answers_B = pd.read_csv('/home/katya/Downloads/output/data/B.csv')

data = {
    'XA': answers_A,
    'XB': answers_B,
    'students_count': answers_A.shape[0],
    'items_count_A': answers_A.shape[1],
    'items_count_B': answers_B.shape[1],
}

# 2PL model
model_3 = pystan.StanModel(file='/home/katya/Downloads/output/models/3model.stan', model_name='model_3')
fit_model_3 = model_3.sampling(data=data,
                               iter=3000,
                               pars=['beta_A', 'beta_B',
                                     'mu_beta_A', 'mu_beta_B',
                                     'sigma_beta_A', 'sigma_beta_B',
                                     'log_lik_A', 'log_lik_B',
                                     'p_A', 'p_B'],
                               n_jobs=1,
                               chains=3)
print(fit_model_3.to_dataframe(['beta_A', 'beta_B',
                                'mu_beta_A', 'mu_beta_B',
                                'sigma_beta_A', 'sigma_beta_B']))

fit_model_3.to_dataframe().to_csv('/home/katya/Downloads/output/3model.csv')

# model comparison
log_lik2_A = fit_model_3.extract()['log_lik_A']
log_lik2_B = fit_model_3.extract()['log_lik_B']
# loo2_A = psisloo(log_lik2_A)
# loo2_B = psisloo(log_lik2_B)
print('LOO = leave-one-out cross-validation')
# print('Model 1PL:', loo1)
# print('Model 2PL:', loo2_A)
# print('Model 2PL:', loo2_B)
