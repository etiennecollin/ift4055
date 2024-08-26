# Greatly inspired from:
# https://github.com/GFNOrg/torchgfn/blob/master/tutorials/notebooks/intro_gfn_continuous_line_simple.ipynb
# https://arxiv.org/abs/2305.14594

import torch
import numpy as np
from tqdm import trange

class AcquisitionTrainer:
    def __init__(self, **kwargs):
        self.env = None

        self.training_steps = kwargs.get("training_steps", 500)
        self.init_state_value = kwargs.get("init_state_value", 0)
        self.state_dim = kwargs.get("state_dim", 2)
        self.hid_dim = kwargs.get("hid_dim", 64)
        self.lr_model = kwargs.get("lr_model", 1e-3)
        self.lr_logz = kwargs.get("lr_logz", 5e-2)
        self.min_policy_std = kwargs.get("min_policy_std", 0.1)
        self.max_policy_std = kwargs.get("max_policy_std", 4.0)
        self.init_explortation_noise = kwargs.get("init_explortation_noise", 5.0)
        self.batch_size = kwargs.get("batch_size", 256)
        self.inference_batch_size = kwargs.get("inference_batch_size", 10_000)
        self.trajectory_length = kwargs.get("trajectory_length", 5)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.forward_model, self.backward_model, self.logZ, self.optimizer = self._create_models()

    def _create_models(self):
        forward_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hid_dim),
            torch.nn.ELU(),
            torch.nn.Linear(self.hid_dim, self.hid_dim),
            torch.nn.ELU(),
            torch.nn.Linear(self.hid_dim, self.state_dim)
        ).to(self.device)

        backward_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.hid_dim),
            torch.nn.ELU(),
            torch.nn.Linear(self.hid_dim, self.hid_dim),
            torch.nn.ELU(),
            torch.nn.Linear(self.hid_dim, self.state_dim)
        ).to(self.device)

        logZ = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        optimizer = torch.optim.Adam(
            [
                {'params': forward_model.parameters(), 'lr': self.lr_model},
                {'params': backward_model.parameters(), 'lr': self.lr_model},
                {'params': [logZ], 'lr': self.lr_logz},
            ]
        )

        return forward_model, backward_model, logZ, optimizer

    def _step(self, x, action):
        new_x = torch.zeros_like(x)
        new_x[:, 0] = x[:, 0] + action  # Add action delta.
        new_x[:, 1] = x[:, 1] + 1  # Increment step counter.

        return new_x

    def _initialize_state(self, batch_size):
        x = torch.zeros((batch_size, self.state_dim), device=self.device)
        x[:, 0] = self.init_state_value

        return x

    def _get_policy_dist(self, model, x, off_policy_noise=0):
        pf_params = model(x)
        policy_mean = pf_params[:, 0]
        policy_std = torch.sigmoid(pf_params[:, 1]) * (self.max_policy_std - self.min_policy_std) + self.min_policy_std
        policy_dist = torch.distributions.Normal(policy_mean, policy_std)
        exploration_dist = torch.distributions.Normal(policy_mean, policy_std + off_policy_noise)

        return policy_dist, exploration_dist

    def set_env(self, env):
        self.env = env

    def inference(self):
        with torch.no_grad():
            trajectory = torch.zeros((self.inference_batch_size, self.trajectory_length + 1, self.state_dim), device=self.device)
            trajectory[:, 0, 0] = self.init_state_value

            x = self._initialize_state(self.inference_batch_size)

            for t in range(self.trajectory_length):
                policy_dist, _ = self._get_policy_dist(self.forward_model, x)
                action = policy_dist.sample()

                new_x = self._step(x, action)
                trajectory[:, t + 1, :] = new_x
                x = new_x

        return trajectory, trajectory[:, -1, 0]

    def train(self, surrogate, oracle, graph_x, graph_y, eval_i, problem_i):
        losses = []
        tbar = trange(self.training_steps)
        exploration_schedule = np.linspace(self.init_explortation_noise, 0,  self.training_steps)

        for it in tbar:
            x = self._initialize_state(self.batch_size)

            # Trajectory stores all of the states in the forward loop.
            trajectory = torch.zeros((self.batch_size, self.trajectory_length + 1, self.state_dim), device=self.device)
            logPF = torch.zeros((self.batch_size,), device=self.device)
            logPB = torch.zeros((self.batch_size,), device=self.device)

            # Forward loop to generate full trajectory and compute logPF.
            for t in range(self.trajectory_length):
                policy_dist, exploration_dist = self._get_policy_dist(self.forward_model, x, exploration_schedule[it])
                action = exploration_dist.sample()
                logPF += policy_dist.log_prob(action)

                new_x = self._step(x, action)
                trajectory[:, t + 1, :] = new_x
                x = new_x

            # Backward loop to compute logPB from existing trajectory under the backward policy.
            for t in range(self.trajectory_length, 2, -1):
                policy_dist, _ = self._get_policy_dist(self.backward_model, trajectory[:, t, :])
                action = trajectory[:, t, 0] - trajectory[:, t - 1, 0]
                logPB += policy_dist.log_prob(action)


            # Get potential new point
            new_x = trajectory[:, -1, 0].mean(dim=0).cpu()
            new_y = oracle(new_x)

            # Add new point to the surrogate and fit it
            surrogate.add_data_point(new_x, new_y)
            surrogate.train()

            # Get the posterior
            posterior = surrogate.get_posterior()
            gp_samples = surrogate.get_samples(posterior, mean=True)
            if it%20 == 0:
                surrogate.plot_gp(graph_x, graph_y, posterior, path=f"./plots/gif_gfn_3/gfn_{problem_i:02d}_{eval_i:04d}_{it:04d}.png")

            reward, a, b, c, kl_diff = self.env.reward(new_x, surrogate.train_x, surrogate.train_y, gp_samples)
            loss = (self.logZ + logPF - logPB - reward).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            surrogate.pop_data_point()
            tbar.set_postfix_str(
                f"i:{it} (l={np.mean(losses[-10:]):.2e}, r={reward:.2e}, t:{[f'{n.item():.2e}' for n in [a, b, c, kl_diff]]})"
            )

        return self.forward_model, self.backward_model, self.logZ

    def save(self, path_forward, path_backward):
        torch.save(self.forward_model.state_dict(), path_forward)
        torch.save(self.backward_model.state_dict(), path_backward)

    def load(self, path_forward, path_backward):
        self.forward_model.load_state_dict(torch.load(path_forward))
        self.backward_model.load_state_dict(torch.load(path_backward))
