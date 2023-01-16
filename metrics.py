import pandas as pd
import numpy as np

df1 = pd.read_csv('/home/katya/Downloads/output/1model.csv')

M = 3
N = 1500


def find_GELMAN_RUBIN_RUBINSHTEIN(seq, sigma):
    average_1 = np.average(seq[:1500])
    average_2 = np.average(A[0][1500:3000])
    average_3 = np.average(A[0][3000:])

    average = np.average([average_1, average_2, average_3])

    sigma_average_1 = np.average(sigma[:1500]) ** 2
    sigma_average_2 = np.average(sigma[1500:3000]) ** 2
    sigma_average_3 = np.average(sigma[3000:]) ** 2

    B_param = N / (M - 1) * sum([(average_1 - average) ** 2,
                                 (average_2 - average) ** 2,
                                 (average_3 - average) ** 2])

    W_param = np.average([sigma_average_3, sigma_average_1, sigma_average_2])

    V_s_kryshychkoy = (N - 1) / N * W_param + (M + 1) / (M * N) * B_param

    return np.sqrt(2 * V_s_kryshychkoy / W_param)


with open('/home/katya/Downloads/output/metrics/GELMAN', 'w') as f:
    sigma_A = df1['sigma_beta_A']
    sigma_B = df1['sigma_beta_B']

    A = [list(df1[f'beta_A[{i}]']) for i in range(1, 9)]
    B = [list(df1[f'beta_B[{i}]']) for i in range(1, 11)]

    f.write("Model 1\n")
    print("Model 1")

    for i, elem in enumerate(A):
        f.write(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}\n")
        print(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}")

    for i, elem in enumerate(B):
        f.write(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}\n")
        print(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}")

    f.write("\n")
    print("\n")
    df2 = pd.read_csv('/home/katya/Downloads/output/2model.csv')

    sigma_A = df2['sigma_beta_A']
    sigma_B = df2['sigma_beta_B']

    A = [list(df2[f'beta_A[{i}]']) for i in range(1, 9)]
    B = [list(df2[f'beta_B[{i}]']) for i in range(1, 11)]

    f.write("Model 2\n")
    print("Model 2")

    for i, elem in enumerate(A):
        f.write(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}\n")
        print(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}")

    for i, elem in enumerate(B):
        f.write(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}\n")
        print(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}")

    f.write("\n")
    print("\n")

    df3 = pd.read_csv('/home/katya/Downloads/output/3model.csv')

    sigma_A = df3['sigma_beta_A']
    sigma_B = df3['sigma_beta_B']

    A = [list(df3[f'beta_A[{i}]']) for i in range(1, 9)]
    B = [list(df3[f'beta_B[{i}]']) for i in range(1, 11)]

    f.write("Model 3\n")
    print("Model 3")

    for i, elem in enumerate(A):
        f.write(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}\n")
        print(f"Metric(A_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_A)}")

    for i, elem in enumerate(B):
        f.write(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}\n")
        print(f"Metric(B_{i + 1}) Gelman-Rubin R`hat - : {find_GELMAN_RUBIN_RUBINSHTEIN(elem, sigma_B)}")

    f.write("\n")
    print("\n")
