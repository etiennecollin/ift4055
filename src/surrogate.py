import numpy as np
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.utils.transforms import normalize, unnormalize

import matplotlib.pyplot as plt
import seaborn as sns

class BayesianGPTrainer:
    def __init__(self, train_x, train_y, graph_x, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.train_x = train_x.to(self.device, self.dtype)
        self.train_y = train_y.to(self.device, self.dtype)
        self.graph_x = graph_x.to(self.device, self.dtype)

        self.x_bounds = kwargs.get("x_bounds", None)
        self.n_posterior_samples = kwargs.get("n_posterior_samples", 1000)
        self.warmup_steps = kwargs.get("warmup_steps", 512)
        self.num_samples = kwargs.get("num_samples", 256)
        self.thinning = kwargs.get("thinning", 32)
        self.noise_scale = kwargs.get("noise_scale", 1e-8)
        self.jit_compile = kwargs.get("jit_compile", True)
        self.save_plot = kwargs.get("save_plot", False)
        self.show_plot = kwargs.get("show_plot", True)
        self.disable_progbar = kwargs.get("disable_progbar", False)

        if self.x_bounds is not None:
            assert type(self.x_bounds) in [tuple, list] and len(self.x_bounds)==2
            self.x_normalizing_bounds = torch.stack([bound*torch.ones(self.train_x.shape[-1]) for bound in self.x_bounds])
        else:
            self.x_normalizing_bounds = None


        # Create model
        self._update_model()

    def _update_model(self):
        self.model = SaasFullyBayesianSingleTaskGP(
            train_X=self.train_x,
            train_Y=self.train_y,
            train_Yvar=torch.full_like(self.train_y, self.noise_scale),
            input_transform=Normalize(self.train_x.shape[-1], bounds=self.x_normalizing_bounds),
            outcome_transform=Standardize(self.train_y.shape[-1]),
        )

    def add_data_point(self, x, y):
        x = x.unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
        y = y.unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
        self.train_x = torch.cat((self.train_x, x), 0).to(self.device, self.dtype)
        self.train_y = torch.cat((self.train_y, y), 0).to(self.device, self.dtype)

        # Update model with new data
        self._update_model()

    def pop_data_point(self):
        self.train_x = self.train_x[:-1]
        self.train_y = self.train_y[:-1]
        self._update_model()

    def train(self):
        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=self.disable_progbar,
            jit_compile=self.jit_compile
        )


    def get_posterior(self):
        with torch.no_grad():
            posterior = self.model.posterior(self.graph_x)
            return posterior

    def get_samples(self, posterior, mean=False):
        with torch.no_grad():
            sample_shape = torch.Size([self.n_posterior_samples])
            posterior_samples = posterior.rsample(sample_shape).mean(dim=0).squeeze(-1).T

            if mean:
                posterior_samples = posterior_samples.mean(dim=1, keepdim=True)

            return posterior_samples

    def plot_gp(self, true_x, true_y, posterior, path=None):
        true_x = true_x.to(self.device, self.dtype)
        true_y = true_y.to(self.device, self.dtype)

        with torch.no_grad():
            # Transpose posterior_samples
            posterior_samples = self.get_samples(posterior).T

            # Get upper and lower confidence bounds (2 standard deviations from the mean)
            lower, upper = posterior.mvn.confidence_region()
            mean_lower = lower.mean(dim=0)
            mean_upper = upper.mean(dim=0)
            # Average across the dimensions to get a single prediction per test point
            mean_predictions = posterior_samples.mean(dim=0)

        # Plot the results
        plt.figure(figsize=(12, 6))

        plt.plot(true_x.cpu(), true_y.cpu(), linewidth=2, label="True Function", color="green")    
        plt.scatter(self.train_x.cpu(), self.train_y.cpu(), label="Training Data", color="black")

        plt.fill_between(true_x.squeeze().cpu(), mean_lower.cpu(), mean_upper.cpu(), alpha=0.2, label="$2\sigma$ Confidence Interval")
        plt.plot(true_x.cpu(), mean_predictions.cpu(), linewidth=2, label="Mean Prediction")

        plt.title("SAASBO Gaussian Process")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(-0.1, 0.5)

        plt.tight_layout()
        plt.legend()
        if self.save_plot:
            if path is None:
                raise ValueError("If `save` is `True`, `path` must be specified including the filename of the saved plot.")
            plt.savefig(path)
        if self.show_plot:
            plt.show()
        else:
            plt.close()
