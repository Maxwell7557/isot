from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')

np.random.seed(1234)
import pystan

answers_A = pd.read_csv('/home/katya/Downloads/output/data/A.csv')
answers_B = pd.read_csv('/home/katya/Downloads/output/data/B.csv')

model_code = """data {
    int<lower=0> students_count;
    int<lower=0> items_count_A;
    int<lower=0> items_count_B;
    int<lower=0, upper=1> XA[students_count, items_count_A];
    int<lower=0, upper=1> XB[students_count, items_count_B];}

parameters {
    vector[students_count] theta_A;
    vector[students_count] theta_B;
    vector<lower=0>[items_count_A] alpha_A;
    vector<lower=0>[items_count_B] alpha_B;
    vector[items_count_A] beta_A;
    vector[items_count_B] beta_B;
    real mu_beta_A;
    real mu_beta_B;
    real<lower=0> sigma_alpha_A;
    real<lower=0> sigma_alpha_B;
    real<lower=0> sigma_beta_A;
    real<lower=0> sigma_beta_B;
}

model {
    theta_A ~ normal(0, 1);
    theta_B ~ normal(0, 1);

    beta_A ~ normal(mu_beta_A, sigma_beta_A);
    mu_beta_A ~ normal(0, 6);
    sigma_beta_A ~ cauchy(0, 8);
    alpha_A ~ lognormal(0, sigma_alpha_A);
    sigma_alpha_A ~ cauchy(0, 8);

    beta_B ~ normal(mu_beta_B, sigma_beta_B);
    mu_beta_B ~ normal(0, 6);
    sigma_beta_B ~ cauchy(0, 10);
    alpha_B ~ lognormal(0, sigma_alpha_B);
    sigma_alpha_B ~ cauchy(0, 10);

    for(i in 1:students_count) {
        for (j in 1:items_count_A) {
            real p;  // create a local variable within the loop to make Stan code more readable
            p = inv_logit(alpha_A[j] * (theta_A[i] - beta_A[j]));
            XA[i, j] ~ bernoulli(p);
        }
        for (j in 1:items_count_B) {
            real p;  // create a local variable within the loop to make Stan code more readable
            p = inv_logit(alpha_B[j] * (theta_B[i] - beta_B[j]));
            XB[i, j] ~ bernoulli(p);
        }
    }
}

generated quantities {
    vector[items_count_A] log_lik_A[students_count];
    vector[items_count_B] log_lik_B[students_count];

    vector[items_count_A] p_A[students_count];
    vector[items_count_B] p_B[students_count];
    for(i in 1:students_count) {
        for (j in 1:items_count_A) {
            real p;
            p = inv_logit(alpha_A[j] * (theta_A[i] - beta_A[j]));
            p_A[i, j] = p;
            log_lik_A[i, j] = bernoulli_lpmf(XA[i, j] | p);
        }
        for (j in 1:items_count_B) {
            real p;
            p = inv_logit(alpha_B[j] * (theta_B[i] - beta_B[j]));
            p_B[i, j] = p;
            log_lik_B[i, j] = bernoulli_lpmf(XB[i, j] | p);
        }
    }
}"""


data = {
    'XA': answers_A,
    'XB': answers_B,
    'students_count': answers_A.shape[0],
    'items_count_A': answers_A.shape[1],
    'items_count_B': answers_B.shape[1],
}

# 2PL model
model_1 = pystan.StanModel(model_code = model_code )
fit_model_1 = model_1.sampling(data=data,
                               iter=3000,
                               pars=['beta_A', 'beta_B',
                                     'mu_beta_A', 'mu_beta_B',
                                     'sigma_beta_A', 'sigma_beta_B',
                                     'log_lik_A', 'log_lik_B',
                                     'p_A', 'p_B'],
                               n_jobs=1,
                               chains=3)
print(fit_model_1.to_dataframe(['beta_A', 'beta_B',
                                'mu_beta_A', 'mu_beta_B',
                                'sigma_beta_A', 'sigma_beta_B']))

df = fit_model_1.to_dataframe()
df.to_csv('/home/katya/Downloads/output/1model.csv')

chain_id = df['draw']

# model comparison
# log_lik2_A = fit_model_1.extract()['log_lik_A']
# log_lik2_B = fit_model_1.extract()['log_lik_B']
# p_A = fit_model_1.extract()['p_A']
# p_B = fit_model_1.extract()['p_B']
# loo2_A = psisloo(log_lik2_A)
# loo2_B = psisloo(log_lik2_B)
# print('LOO = leave-one-out cross-validation')
# print('Model 1PL:', loo1)
# print('Model 2PL:', loo2_A)
# print('Model 2PL:', loo2_B)
#
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.hist(trace['mu'][:], 25, histtype='step')
# plt.subplot(1, 2, 2)
# plt.hist(trace['sigma'][:], 25, histtype='step')
