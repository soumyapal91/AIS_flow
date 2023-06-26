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

    def __init__(self, args):
        self.particles = list()
        self.logW = list()

        if args.algorithm == 'NF-PMC':
            self.nf_model = NormalizingFlowModel(args)
            self.optimizer = torch.optim.RMSprop(self.nf_model.parameters(), lr=args.lr_nf)
            if args.use_cuda:
                self.nf_model.to(args.device)

    def sample_n_weight_nf(self, current_prop, args):
        # sampling
        samples = current_prop.mean[:, None, :] + np.array([np.dot(scipy.linalg.cholesky(current_prop.cov[:, :, n], lower=True),
                                   np.random.normal(size=[args.dim, args.K])).T for n in range(args.N)])

        samples = torch.from_numpy(samples).to(args.device)

        self.optimizer.zero_grad()
        self.nf_model.train()

        samples, log_w = self.nf_model(samples, current_prop)

        return samples, log_w

    def adaptation_step_nf(self, particles, logW, current_prop, args):
        loss = -1.0 * torch.mean(logW - torch.max(logW, dim=1, keepdim=True)[0])
        # loss = torch.mean(torch.square(torch.exp(logW - torch.max(logW, dim=1, keepdim=True)[0])))
        loss.backward()
        self.optimizer.step()

        particles = particles.cpu().detach().numpy()
        logW = logW.cpu().detach().numpy()

        current_prop = resample_adapt(particles, logW, current_prop, args)

        return particles, logW, current_prop


def AIS_main(args):

    args = Initialization(args)
    current_prop = CurrentProp(args)
    output = Output(args)

    if args.algorithm == 'VAPIS':
        G_mu = np.zeros(current_prop.mean.shape)  # running estimate of squared norm of grad, useful only for VAPIS

    t_start = time.time()
    args.runtime = list()
    for _ in tqdm(range(args.J)):

        if args.algorithm == 'NF-PMC':
            # sampling and weighting
            samples, log_w = output.sample_n_weight_nf(current_prop, args)

            # adaptation
            samples, log_w, current_prop = output.adaptation_step_nf(samples, log_w, current_prop, args)

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

            elif args.adaptation == 'VAPIS':
                current_prop, G_mu = vi_adapt(samples, log_w, current_prop, args, G_mu)

        output.particles.append(samples)
        output.logW.append(log_w)
        # compute runtime
        args.runtime.append(time.time() - t_start)

    output.particles = np.array(output.particles)
    output.logW = np.array(output.logW)

    return output, args
