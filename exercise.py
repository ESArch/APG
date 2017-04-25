import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


np.random.seed(123)
M = 100

mu = [-6, 2]
sigma = [2, 2]

pi_true = [0.1, 0.9]

x1 = np.random.normal(mu[0], sigma[0], M*pi_true[0])
x2 = np.random.normal(mu[1], sigma[1], M*pi_true[1])

xs = np.concatenate((x1, x2))
ys = mlab.normpdf(xs, mu[0], sigma[0]) * pi_true[0] + mlab.normpdf(xs, mu[1], sigma[1]) * pi_true[1]

fig = plt.figure()
# plt.hist(xs, 25, normed=1, facecolor='g', alpha=0.75)

x_aux = np.linspace(-15, 10, 100)

plt.plot( x_aux, mlab.normpdf(x_aux,mu[0], sigma[0]) * pi_true[0], color="r", linestyle="-", linewidth=2)
plt.plot( x_aux, mlab.normpdf(x_aux,mu[1], sigma[1]) * pi_true[1], color="r", linestyle="-", linewidth=2)

plt.scatter(xs, ys)








pi_guess = [0.7, 0.3]
K = len(pi_guess)

plt.plot(x_aux, mlab.normpdf(x_aux, mu[0], sigma[0]) * pi_guess[0], color="black", linestyle="--", linewidth=1)
plt.plot(x_aux, mlab.normpdf(x_aux, mu[1], sigma[1]) * pi_guess[1], color="black", linestyle="--", linewidth=1)

while True:

    old_pi = pi_guess.copy()

    zs = np.zeros((K,M))
    for k in range(K):
        for m in range(M):
            # p = norm.pdf(xs[m], mu[k], sigma[k])

            p = norm(mu[k], sigma[k]).pdf(xs[m])
            zs[k][m] = pi_guess[k] * p

    zs /= zs.sum(0)

    pi_guess = np.zeros(K)
    for k in range(K):
        for m in range(M):
            pi_guess[k] += zs[k][m]
    pi_guess /= M


    print(pi_guess)

    plt.plot(x_aux, mlab.normpdf(x_aux, mu[0], sigma[0]) * pi_guess[0], color="black", linestyle="--", linewidth=1)
    plt.plot(x_aux, mlab.normpdf(x_aux, mu[1], sigma[1]) * pi_guess[1], color="black", linestyle="--", linewidth=1)

    error = sum([abs(old_pi[k] - pi_guess[k]) for k in range(2)])
    if error < 0.001:
        break


plt.plot(x_aux, mlab.normpdf(x_aux, mu[0], sigma[0]) * pi_guess[0], color="black", linestyle="--", linewidth=3)
plt.plot(x_aux, mlab.normpdf(x_aux, mu[1], sigma[1]) * pi_guess[1], color="black", linestyle="--", linewidth=3)

fig.savefig("original_values.png")


