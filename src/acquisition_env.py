# Greatly inspired from:
# https://github.com/GFNOrg/torchgfn/blob/master/tutorials/notebooks/intro_gfn_continuous_line_simple.ipynb
# https://arxiv.org/abs/2305.14594

import torch
import torch.nn as nn
import numpy as np
from pqm import pqm_chi2

class AcqEnvironment():
    def __init__(self, max_samples, test_samples):
        self.prev_gp_samples = None
        self.max_samples = max_samples if max_samples is not None else 10_000
        self.test_samples = test_samples.cpu()
        self.pqm_num_refs = self.test_samples.shape[0] // 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def update_prev_posterior(self, gp_samples):
        self.prev_gp_samples = gp_samples

    def reward(self, new_x, train_x, train_y, gp_samples):
        kl_diff = torch.tensor(0.0)
        i = train_y.shape[0]
        if self.prev_gp_samples is not None:
            # This makes sure that the shape of the gp_samples is the same as that of the train_y
            indices_1 = torch.randperm(self.max_samples)[:i]
            indices_2 = torch.randperm(self.max_samples)[:i]
            prev_gp_samples = self.prev_gp_samples[indices_1]
            gp_samples = gp_samples[indices_2]

            # If we are improving, KL of prev_gp_samples > gp_samples
            # i.e. kl_1 > kl_2
            kl_1 = self.kl_loss(prev_gp_samples, train_y)
            kl_2 = self.kl_loss(gp_samples, train_y)
            kl_diff = nn.functional.sigmoid(kl_1-kl_2)

        chi2_mean_normalizer = lambda x: 1 / (0.5 * np.abs(np.mean(x) - (self.pqm_num_refs-1)) + 1)
        chi2_var_normalizer = lambda x: 1 / (0.5 * np.abs(np.std(x) - np.sqrt(2*(self.pqm_num_refs-1))) + 1)

        # PQMass between gp_samples and test dataset
        chi2_1 = pqm_chi2(gp_samples.cpu(), self.test_samples, num_refs=self.pqm_num_refs, re_tessellation=50)
        pqmass_1a = chi2_mean_normalizer(chi2_1)
        pqmass_1b = chi2_var_normalizer(chi2_1)
        pqmass_1 = (pqmass_1a + pqmass_1b)/2


        # PQMass between train and test dataset
        chi2_2 = pqm_chi2(train_y.cpu(), self.test_samples, num_refs=self.pqm_num_refs, re_tessellation=50)
        pqmass_2a = chi2_mean_normalizer(chi2_2)
        pqmass_2b = chi2_var_normalizer(chi2_2)
        pqmass_2 = (pqmass_2a + pqmass_2b)/2

        a = torch.tensor(1 - i/self.max_samples)
        b = torch.tensor(pqmass_1, requires_grad=True)
        c = torch.tensor(pqmass_2, requires_grad=True)
        d = torch.tensor(1) if new_x not in train_x[:-1] else torch.tensor(0)
        reward = (a + b + c + kl_diff)/4
        if new_x in train_x[:-1]:
            reward *= 1e-4
        reward = torch.log(reward).to(self.device)
        return reward, a, b, c, kl_diff
