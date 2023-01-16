import pandas as pd
import pystan

answers_A = pd.read_csv('/home/maxwell/Downloads/output/data/A.csv')
answers_B = pd.read_csv('/home/maxwell/Downloads/output/data/B.csv')

data = {
    'XA': answers_A,
    'XB': answers_B,
    'students_count': answers_A.shape[0],
    'items_count_A': answers_A.shape[1],
    'items_count_B': answers_B.shape[1],
}

# 2PL model
model_2 = pystan.StanModel(file='/home/maxwell/Downloads/output/models/2model.stan', model_name='model_2')
fit_model_2 = model_2.sampling(data=data,
                               iter=3000,
                               pars=['beta_A', 'beta_B',
                                     'mu_beta_A', 'mu_beta_B',
                                     'sigma_beta_A', 'sigma_beta_B',
                                     'log_lik_A', 'log_lik_B',
                                     'p_A', 'p_B'],
                               n_jobs=1,
                               chains=3)
print(fit_model_2.to_dataframe(['beta_A', 'beta_B',
                                'mu_beta_A', 'mu_beta_B',
                                'sigma_beta_A', 'sigma_beta_B']))

fit_model_2.to_dataframe().to_csv('/home/maxwell/Downloads/output/2model.csv')

# model comparison
log_lik2_A = fit_model_2.extract()['log_lik_A']
log_lik2_B = fit_model_2.extract()['log_lik_B']
print('LOO = leave-one-out cross-validation')
