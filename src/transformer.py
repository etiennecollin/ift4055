import torch
import torch.nn as nn
import numpy as np
from transformer_models import *
from tqdm import trange
import matplotlib.pyplot as plt


class SetTransformerTrainer:
    def __init__(self, dim_input, dim_hidden, dim_output, **kwargs):
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.num_outputs = kwargs.get("num_outputs", 8)
        self.num_inds = kwargs.get("num_inds", 64)
        self.num_heads = kwargs.get("num_heads", 8)

        self.lr = kwargs.get("lr", 1e-12)
        self.lr_multiplicator = kwargs.get("lr_multiplicator", 1e-2)
        self.tensor_length_min = kwargs.get("tensor_length_min", 2**2)
        self.tensor_length_max = kwargs.get("tensor_length_max", 2**15)
        self.cosine_loss = kwargs.get("cosine_loss", False)
        self.use_deepset = kwargs.get("use_deepset", False)
        self.dtype = kwargs.get("dtype", torch.float32)

        # Present to fix issue with dtypes in st_modules
        torch.set_default_dtype(self.dtype)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = (
            nn.CosineEmbeddingLoss(reduction="sum").to(device)
            if self.cosine_loss
            else nn.PairwiseDistance(p=2, eps=0).to(self.device)
        )

        # Scales the L2Norm between 0 and 1.
        # 1 when the L2Norm is 0
        # 0 when the L2Norm approaches infinity
        self.scale_factor = lambda x: (-x) / (1 + torch.abs(x)) + 1

    def _create_model(self):
        # Initialize the model based on the given parameters
        if self.use_deepset:
            return DeepSet(self.dim_input, self.num_outputs, self.dim_output, self.dim_hidden)
        else:
            return SetTransformer(
                self.dim_input, self.num_outputs, self.dim_output, self.num_inds, self.dim_hidden, self.num_heads
            )

    def gen_data(self, number_pts, permutation=True, domain=(0, 100)):
        rng = np.random.default_rng()
        x = rng.uniform(domain[0], domain[1], (number_pts, self.dim_input))

        if permutation:
            y = rng.permutation(x)
        else:
            y = rng.uniform(domain[0], domain[1], (number_pts, self.dim_input))

        x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        x, y = torch.from_numpy(x).to(self.device, self.dtype), torch.from_numpy(y).to(self.device, self.dtype)

        return x, y

    def train(self, iterations):
        losses = []
        losses_valid_perm = []
        losses_invalid_perm = []

        print("_i: L2Norm of Invalid Permutations")
        print("_v: L2Norm of Valid Permutations")
        print("loss: Loss of model")

        tbar = trange(iterations, desc="Training", unit="it")
        for i in tbar:
            # Update learning rate for second half of training
            if i == iterations // 2:
                self.optimizer.param_groups[0]["lr"] *= self.lr_multiplicator

            # Get a random tensor length
            rng = np.random.default_rng()
            points_per_tensor = rng.integers(self.tensor_length_min, self.tensor_length_max)
            # Get 2x two permutations of the same list
            x1, y1 = self.gen_data(points_per_tensor)
            x2, y2 = self.gen_data(points_per_tensor)

            # Prepare the pairs for the loss
            x1, y1, x2, y2 = self.model(x1), self.model(y1), self.model(x2), self.model(y2)
            if self.cosine_loss:
                a = torch.cat((x1, x2, x1, x1, x2, y1), 0)
                b = torch.cat((y1, y2, y2, x2, y1, y2), 0)
                c = torch.tensor([1, 1, -1, -1, -1, -1], dtype=torch.int).to(self.device)
                loss = self.criterion(a, b, c)

                # For logging
                tmp_1a = torch.cat((x1, x2), 0)
                tmp_1b = torch.cat((y1, y2), 0)
                tmp_1c = torch.tensor([1, 1], dtype=torch.int).to(self.device)
                loss_valid_perm = self.criterion(tmp_1a, tmp_1b, tmp_1c).item()
                tmp_2a = torch.cat((x1, x1, x2, y1), 0)
                tmp_2b = torch.cat((y2, x2, y1, y2), 0)
                tmp_2c = torch.tensor([-1, -1, -1, -1], dtype=torch.int).to(self.device)
                loss_invalid_perm = self.criterion(tmp_2a, tmp_2b, tmp_2c).item()
            else:
                a = self.criterion(x1, y1)
                b = self.criterion(x2, y2)
                c = self.criterion(x1, y2)
                d = self.criterion(x1, x2)
                e = self.criterion(x2, y1)
                f = self.criterion(y1, y2)

                loss_a = a + b
                loss_b = self.scale_factor(c + d + e + f)
                loss = loss_a + loss_b

                loss_valid_perm = loss_a.item()
                loss_invalid_perm = loss_b.item()

            # Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            losses_valid_perm.append(loss_valid_perm)
            losses_invalid_perm.append(loss_invalid_perm)
            tbar.set_postfix_str(
                "loss={:.3E}, v={:.3E}, i={:.3E}".format(
                    np.mean(losses), np.mean(losses_valid_perm), np.mean(losses_invalid_perm)
                )
            )

        return losses, losses_valid_perm, losses_invalid_perm

    def test(self, iterations=10, perm_threshold=1e-12, only_valid_perms=False, no_break=False):
        with torch.no_grad():
            l2norm = torch.nn.PairwiseDistance(p=2, eps=0)
            rng = np.random.default_rng()

            print("Testing...")
            valid_list = []
            invalid_list = []
            for i in range(only_valid_perms, 2):
                tbar = trange(
                    iterations,
                    desc="Testing invalid permutations" if i == 0 else "Testing valid permutations",
                    unit="it",
                )
                for j in tbar:
                    # Get a random tensor length
                    points_per_tensor = rng.integers(self.tensor_length_min, self.tensor_length_max)
                    # Get two lists
                    x1, x2 = self.gen_data(points_per_tensor, permutation=i)

                    # Get embeddings
                    embedding_1 = self.model(x1).squeeze(0)
                    embedding_2 = self.model(x2).squeeze(0)

                    # Compute distance
                    distance = l2norm(embedding_1, embedding_2).item()
                    if (
                        (i == 0 and distance <= perm_threshold) or (i == 1 and distance > perm_threshold)
                    ) and not no_break:
                        print(distance)
                        print(x1.shape, x1)
                        print(x2.shape, x2)
                        print(embedding_1.shape, embedding_1)
                        print(embedding_2.shape, embedding_2)
                        break

                    if i == 0:
                        invalid_list.append(distance)
                    else:
                        valid_list.append(distance)

        return valid_list, invalid_list

    def plot(self, valid_perm_distances, invalid_perm_distances, show_plot=True, save_path=None):
        # Create a new figure
        plt.figure(figsize=(10, 3))

        # Plot valid_list on a fixed y-value (e.g., y=1)
        plt.scatter(
            valid_perm_distances, [0] * len(valid_perm_distances), color="g", label="Permutations", alpha=0.6, s=10
        )

        # Plot invalid_list on a different fixed y-value (e.g., y=0)
        plt.scatter(
            invalid_perm_distances,
            [0] * len(invalid_perm_distances),
            color="r",
            label="Non-Permutations",
            alpha=0.6,
            s=10,
        )

        # Add title and labels
        plt.title("Embeddings of Set Transformer")
        plt.xlabel("L2 Norm of Distance Between Embeddings Pairs")
        plt.xscale("log")
        plt.yticks([])  # Remove y-axis ticks
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        else:
            plt.close()

        # Show the plot
        plt.show()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
