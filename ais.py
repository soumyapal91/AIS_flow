import numpy as np
import scipy, torch, random
import time
from tqdm import tqdm
from utils import *
from utils_torch import *
from target_dist import *
from initialization import *
from proposal import *
from adapt import *


class Output:

    def __init__(self, args, current_prop):
        self.particles = list()
        self.logW = list()

        if args.algorithm == 'NF-PMC' or args.algorithm == 'NF-PMC2':
            self.nf_model = NormalizingFlowModel(args, current_prop)
            self.optimizer = torch.optim.RMSprop(self.nf_model.parameters(), lr=args.lr_nf, alpha=0.9, eps=1e-8)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_nf, gamma=args.gamma)
            if args.use_cuda:
                self.nf_model.to(args.device)

    def sample_n_weight_nf(self, args):
        # sampling

        if args.learn_var:
            samples = np.random.normal(size=[args.N, args.K, args.dim])
        else:
            samples = args.sigma_prop * np.random.normal(size=[args.N, args.K, args.dim])

        samples = torch.from_numpy(samples).to(args.device)

        self.optimizer.zero_grad()
        self.nf_model.train()

        samples, log_target, log_proposal = self.nf_model(samples)

        return samples, log_target, log_proposal

    def adaptation_step_nf(self, args, particles, log_target, log_proposal):

        if args.loss == 'KL':
            loss = -torch.mean(log_target - log_proposal)
        elif args.loss == 'KLrev':
            log_w_ = (log_target - log_proposal).detach()
            weight = torch.exp(log_w_ - torch.max(log_w_))
            weight = weight / torch.sum(weight)
            loss = -torch.mean(weight * log_proposal)
        elif args.loss == 'div2':
            log_w_ = 2 * (log_target - log_proposal).detach()
            weight = torch.exp(log_w_ - torch.max(log_w_))
            weight = weight / torch.sum(weight)
            loss = -torch.mean(weight * log_proposal)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        logW = log_target - log_proposal

        particles = particles.cpu().detach().numpy()
        logW = logW.cpu().detach().numpy()

        return particles, logW


def AIS_main(args):
    args = init_params(args)
    args = init_alg(args)
    current_prop = CurrentProp(args)
    output = Output(args, current_prop)

    if args.algorithm == 'VAPIS':
        G_mu = np.zeros(current_prop.mean.shape)  # running estimate of squared norm of grad, useful only for VAPIS

    t_start = time.time()
    args.runtime = list()
    for j in tqdm(range(args.J)):

        if args.algorithm == 'NF-PMC' or args.algorithm == 'NF-PMC2':
            # sampling and weighting
            samples, log_target_, log_proposal_ = output.sample_n_weight_nf(args)

            # adaptation
            samples, log_w = output.adaptation_step_nf(args, samples, log_target_, log_proposal_)

        else:
            # sampling
            samples = proposal_step(current_prop, args)

            # weighting
            log_w = importance_weighting(samples, current_prop, args)

            # adaptation
            if args.adaptation == 'Resample':
                current_prop = resample_adapt(samples, log_w, current_prop, args)

            elif args.adaptation == 'Langevin':
                current_prop = langevin_adapt(samples, log_w, current_prop, args)

            elif args.adaptation == 'Newton':
                current_prop = newton_adapt(samples, log_w, current_prop, args)

            elif args.adaptation == 'GRAMIS':
                current_prop = gramis_adapt(samples, log_w, current_prop, args, j)

            elif args.adaptation == 'VAPIS':
                current_prop, G_mu = vi_adapt(samples, log_w, current_prop, args, G_mu)

            elif args.adaptation == 'HMC':
                if j == 0:
                    current_prop = hmc_adapt(current_prop, args, burnin=True)
                else:
                    current_prop = hmc_adapt(current_prop, args)

        output.particles.append(samples)
        output.logW.append(log_w)
        # compute runtime
        args.runtime.append(time.time() - t_start)

    output.particles = np.array(output.particles)
    output.logW = np.array(output.logW)

    return output, args
