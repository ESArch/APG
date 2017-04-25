import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.lines as mlines
from scipy.stats import norm


# Log-likelihood computation
def compute_ll(xs, mu, sigma, pi):
    ll = 0
    for m in range(len(xs)):
        aux = 0
        for k in range(len(mu)):
            aux += pi[i] * norm(mu[k], sigma[k]).pdf(xs[m])
        ll += np.log(aux)

    return ll



np.random.seed(123)
M = 100 # number of samples
epsilon = 0.001 # Convergence criteria


# Original parameters
mu = [-6, 2]
sigma = [2, 2]
pi_true = [0.4, 0.6]


# Data generation
x1 = np.random.normal(mu[0], sigma[0], M * pi_true[0])
x2 = np.random.normal(mu[1], sigma[1], M * pi_true[1])
xs = np.concatenate((x1, x2))


# Samples probability
ys = mlab.normpdf(xs, mu[0], sigma[0]) * pi_true[0] + mlab.normpdf(xs, mu[1], sigma[1]) * pi_true[1]


# Crate the chart
fig = plt.figure()


# Draw the original distributions
x_aux = np.linspace(-15, 10, 100)

plt.plot( x_aux, mlab.normpdf(x_aux,mu[0], sigma[0]) * pi_true[0], color="r", linestyle="-", linewidth=2)
plt.plot( x_aux, mlab.normpdf(x_aux,mu[1], sigma[1]) * pi_true[1], color="r", linestyle="-", linewidth=2)


# Draw the samples
plt.scatter(xs, ys, label="samples", color="g")





pi_guess = [0.7, 0.3] # Initial values of the parameters to stimate
K = len(pi_guess) # number of gaussians in the mixture


# Draw the distributions with the initial values
for i in range(K):
    plt.plot(x_aux, mlab.normpdf(x_aux, mu[i], sigma[i]) * pi_guess[i], color="black", linestyle="--", linewidth=1)
# plt.plot(x_aux, mlab.normpdf(x_aux, mu[1], sigma[1]) * pi_guess[1], color="black", linestyle="--", linewidth=1)



old_ll = compute_ll(xs, mu, sigma, pi_guess)


# EM Algorithm
while True:


    # E Step
    zs = np.zeros((K,M))
    for k in range(K):
        for m in range(M):
            # p = norm.pdf(xs[m], mu[k], sigma[k])

            p = norm(mu[k], sigma[k]).pdf(xs[m])
            zs[k][m] = pi_guess[k] * p

    zs /= zs.sum(0)


    # M Step
    pi_guess = np.zeros(K)
    for k in range(K):
        for m in range(M):
            pi_guess[k] += zs[k][m]
    pi_guess /= M


    print(pi_guess)


    # Draw the distributions obtained in the current iteration
    for i in range(K):
        guess_label = plt.plot(x_aux, mlab.normpdf(x_aux, mu[i], sigma[i]) * pi_guess[i], color="black", linestyle="--", linewidth=1, label = "EM")


    # If convergence stop
    new_ll = compute_ll(xs, mu, sigma, pi_guess)
    if abs(new_ll - old_ll) < epsilon:
        break

    old_ll = new_ll

# Draw the distributions obtained with a thicker line
for i in range(K):
    final_label = plt.plot(x_aux, mlab.normpdf(x_aux, mu[i], sigma[i]) * pi_guess[i], color="black", linestyle="--", linewidth=3, label="final")


# Legend
sample_label = mlines.Line2D([], [], color='g', linestyle="", linewidth=1, marker=".", markersize="10", label="samples")
original_label = mlines.Line2D([], [], color='r', linestyle="-", linewidth=2, label="original")
original_label = mlines.Line2D([], [], color='r', linestyle="-", linewidth=2, label="original")
em_label = mlines.Line2D([], [], color='black', linestyle="--", linewidth=1, label="EM")
em_final_label = mlines.Line2D([], [], color='black', linestyle="--", linewidth=3, label="EM final")


plt.legend(handles=[sample_label, original_label, em_label, em_final_label])
fig.savefig("base_exercise.png")


