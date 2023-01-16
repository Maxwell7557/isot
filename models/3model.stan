data {
    int<lower=0> students_count;
    int<lower=0> items_count_A;
    int<lower=0> items_count_B;
    int<lower=0, upper=1> XA[students_count, items_count_A];
    int<lower=0, upper=1> XB[students_count, items_count_B];
}

parameters {
    vector[students_count] theta_A;
    vector[students_count] theta_B;

    vector[students_count] eta_A;
    vector[students_count] eta_B;

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

    real sigma_error;
    real<lower=0> error;
    real a;
}

model {
    theta_A ~ normal(0, 1);
    sigma_error ~ cauchy(0, 8);
    error ~ normal(0, sigma_error);
    a ~ uniform(-100, 100);

    for (i in 1:students_count) {
        real b;
        b = a * theta_A[i];
        theta_B[i] ~ normal(b, sqrt(a * a + error * error));
    }

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
}

