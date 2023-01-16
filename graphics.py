import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iters = [j for j in range(1500)]


def create_graphics(seq, title, model_name, ):
    for i, elem in enumerate(seq):
        plt.figure(figsize=(10, 4))

        plt.suptitle(f'{model_name}_{title}{i + 1}')

        plt.subplot(1, 2, 1)
        plt.plot(iters, elem[:1500],
                 color=(np.random.random(), np.random.random(), np.random.random()))
        plt.plot(iters, elem[1500:3000],
                 color=(np.random.random(), np.random.random(), np.random.random()))
        plt.plot(iters, elem[3000:],
                 color=(np.random.random(), np.random.random(), np.random.random()))
        plt.title('traceplot')

        plt.subplot(1, 2, 2)
        plt.acorr(elem[:1500],
                  color=(np.random.random(), np.random.random(), np.random.random()))
        plt.acorr(elem[1500:3000],
                  color=(np.random.random(), np.random.random(), np.random.random()))
        plt.acorr(elem[3000:],
                  color=(np.random.random(), np.random.random(), np.random.random()))
        plt.title('autocorrelation')

        plt.savefig(f'graphics/{model_name}/graphic_{i + 1}_{title}.png')
        plt.show()


df1 = pd.read_csv('/home/katya/Downloads/output/1model.csv')
df2 = pd.read_csv('/home/katya/Downloads/output/2model.csv')
df3 = pd.read_csv('/home/katya/Downloads/output/3model.csv')

betas_A = [list(df1[f'beta_A[{i}]']) for i in range(1, 9)]
betas_B = [list(df1[f'beta_B[{i}]']) for i in range(1, 11)]

create_graphics(betas_A, "betas_A", model_name='model_1')
create_graphics(betas_B, "betas_B", model_name='model_1')

betas_A = [list(df2[f'beta_A[{i}]']) for i in range(1, 9)]
betas_B = [list(df2[f'beta_B[{i}]']) for i in range(1, 11)]

create_graphics(betas_A, "betas_A", model_name='model_2')
create_graphics(betas_B, "betas_B", model_name='model_2')

betas_A = [list(df3[f'beta_A[{i}]']) for i in range(1, 9)]
betas_B = [list(df3[f'beta_B[{i}]']) for i in range(1, 11)]

create_graphics(betas_A, "betas_A", model_name='model_3')
create_graphics(betas_B, "betas_B", model_name='model_3')
