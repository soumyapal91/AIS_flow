import numpy as np
import time
from scipy.linalg import lu, cholesky, solve, solve_triangular, sqrtm
from utils import *
from ais import *
from target_dist import *


class Result:

    def __init__(self, output, args):
        self.runtime = args.runtime

        if args.example == 'Gaussian':
            self.mu_true = args.target_mu
            self.m2_true = np.diag(args.target_cov) + (args.target_mu ** 2)

            self.mu_est = list()
            self.m2_est = list()

            self.mse_mu = list()
            self.mse_m2 = list()

            if args.step_by_step:
                for j in range(args.J):
                    particles_ = output.particles[0:j + 1, :, :, :].reshape([-1, args.dim])
                    logW_ = output.logW[0:j + 1, :, :].reshape(-1)

                    self.mu_est.append(particle_estimate(particles_, logW_))
                    self.m2_est.append(particle_estimate(particles_ ** 2, logW_))

                    self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))
                    self.mse_m2.append(MSE(self.m2_est[-1], self.m2_true))
            else:
                particles_ = output.particles.reshape([-1, args.dim])
                logW_ = output.logW.reshape(-1)

                self.mu_est.append(particle_estimate(particles_, logW_))
                self.m2_est.append(particle_estimate(particles_ ** 2, logW_))

                self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))
                self.mse_m2.append(MSE(self.m2_est[-1], self.m2_true))

            print('Runtime: ' + str(self.runtime))
            print('MSE of mean estimate: ' + str(self.mse_mu))
            print('MSE of 2nd momemnt estimate: ' + str(self.mse_m2))

        elif args.example == 'Banana':
            self.mu_true = np.zeros(args.dim)

            self.mu_est = list()

            self.mse_mu = list()

            if args.step_by_step:
                for j in range(args.J):
                    particles_ = output.particles[0:j + 1, :, :, :].reshape([-1, args.dim])
                    logW_ = output.logW[0:j + 1, :, :].reshape(-1)

                    self.mu_est.append(particle_estimate(particles_, logW_))

                    self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))
            else:
                particles_ = output.particles.reshape([-1, args.dim])
                logW_ = output.logW.reshape(-1)

                self.mu_est.append(particle_estimate(particles_, logW_))

                self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))

            print('Runtime: ' + str(self.runtime))
            print('MSE of mean estimate: ' + str(self.mse_mu))

        elif args.example == 'GMM':
            self.mu_true = np.dot(args.target_mu.T, args.target_alpha)
            self.m2_true = np.dot(np.array([(np.diag(args.target_cov[:, :, i]) + (args.target_mu[i, :] ** 2))
                                            for i in range(args.target_nc)]).T, args.target_alpha)

            self.mu_est = list()
            self.m2_est = list()

            self.mse_mu = list()
            self.mse_m2 = list()

            if args.step_by_step:
                for j in range(args.J):
                    particles_ = output.particles[0:j + 1, :, :, :].reshape([-1, args.dim])
                    logW_ = output.logW[0:j + 1, :, :].reshape(-1)

                    self.mu_est.append(particle_estimate(particles_, logW_))
                    self.m2_est.append(particle_estimate(particles_ ** 2, logW_))

                    self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))
                    self.mse_m2.append(MSE(self.m2_est[-1], self.m2_true))
            else:
                particles_ = output.particles.reshape([-1, args.dim])
                logW_ = output.logW.reshape(-1)

                self.mu_est.append(particle_estimate(particles_, logW_))
                self.m2_est.append(particle_estimate(particles_ ** 2, logW_))

                self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true))
                self.mse_m2.append(MSE(self.m2_est[-1], self.m2_true))

            print('Runtime: ' + str(self.runtime))
            print('MSE of mean estimate: ' + str(self.mse_mu))
            print('MSE of 2nd momemnt estimate: ' + str(self.mse_m2))

        elif args.example == 'Logistic':
            self.mu_true = args.w_true

            self.mu_est = list()

            self.mse_mu = list()

            self.ll_test = list()

            if args.step_by_step:
                for j in range(args.J):
                    particles_ = output.particles[0:j + 1, :, :, :].reshape([-1, args.dim])
                    logW_ = output.logW[0:j + 1, :, :].reshape(-1)

                    self.mu_est.append(particle_estimate(particles_, logW_))

                    self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true) / np.mean(self.mu_true ** 2))

                    prob = 1.0 / (1 + np.exp(-np.dot(args.X_test, particles_.T)))
                    prob = np.clip(prob, 1e-14, 1 - 1e-14)
                    labels = np.tile(args.Y_test[None, :].T, [1, particles_.shape[0]])
                    ll_ = np.mean(labels * np.log(prob) + (1 - labels) * np.log(1 - prob), axis=0)[:, None]
                    self.ll_test.append(particle_estimate(ll_, logW_)[0])

            else:
                particles_ = output.particles.reshape([-1, args.dim])
                logW_ = output.logW.reshape(-1)

                self.mu_est.append(particle_estimate(particles_, logW_))

                self.mse_mu.append(MSE(self.mu_est[-1], self.mu_true) / np.mean(self.mu_true ** 2))

                prob = 1.0 / (1 + np.exp(-np.dot(args.X_test, particles_.T)))
                prob = np.clip(prob, 1e-14, 1 - 1e-14)
                labels = np.tile(args.Y_test[None, :].T, [1, particles_.shape[0]])
                ll_ = np.mean(labels * np.log(prob) + (1 - labels) * np.log(1 - prob), axis=0)[:, None]
                self.ll_test.append(particle_estimate(ll_, logW_)[0])

            print('Runtime: ' + str(self.runtime[-1]))
            print('rMSE of weight estimate: ' + str(self.mse_mu))
            print('Log-likelihood of test data: ' + str(self.ll_test))
