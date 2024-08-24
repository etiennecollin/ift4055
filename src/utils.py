import numpy as np
import torch
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()

    def random_nxm_tensor(self, n, m, min_val, max_val):
        # Create a tensor with n points, each having m coordinates
        random_tensor = torch.rand(n, m)
        
        # Scale and shift to the range [min_val, max_val]
        scaled_tensor = random_tensor * (max_val - min_val) + min_val
        
        return scaled_tensor

    def random_gaussian_mixture_pdf(self):
        """
        Returns a lambda function representing the PDF of a Gaussian mixture model
        with two Gaussians where the means are opposite in sign and the standard deviation is randomly generated.
        
        Parameters:
        - None (parameters are generated within the function)
        
        Returns:
        - A lambda function representing the PDF of the Gaussian mixture model
        """    
        # Randomly generate the means and standard deviation
        mu = self.rng.normal(7.0, 1.5)  # Random mean for first Gaussian
        sigma1 = self.rng.lognormal(0.0, 0.5)  # Random standard deviation (log-normal to ensure positivity)
        sigma2 = self.rng.lognormal(0.0, 0.5)  # Random standard deviation (log-normal to ensure positivity)
    
        # Means are opposite in sign
        mu1 = mu
        mu2 = -mu
        
        weight = 0.5  # Equal weight for simplicity (can be adjusted or made random too)
        
        # Define the Gaussian PDF function
        def gaussian_pdf(x, mean, sigma):
            return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((x - mean) / sigma)**2)
        
        # Return the lambda function for the mixture PDF
        return lambda x: (weight * gaussian_pdf(x, mu1, sigma1) +
                          weight * gaussian_pdf(x, mu2, sigma2))

    def plot(self, function, x_values=np.linspace(-10, 10, 1000)):
        # Compute the PDF values
        function_values = [function(x) for x in x_values]
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, function_values, label='Gaussian Mixture PDF', color='blue')
        plt.title('Gaussian Mixture Model PDF')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
