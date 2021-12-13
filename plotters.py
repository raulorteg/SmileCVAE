import numpy as np
import matplotlib.pyplot as plt

class OnlineVariance(object):
    def __init__(self, ndim, iterable=None, ddof=1):
        """
        Welford's algorithm computes the sample variance incrementally.
        Computes the variance of an array online. Memory efficient.

        :param ndim: number of dimensions in the array
        :type ndim: int
        """
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, np.zeros(ndim), np.zeros(ndim)
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        """
        :param datum: n_dim dimnesional array
        :type datum: numpy.ndarray
        """
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)

class LatentPlotter:
    def __init__(self, ndim, file_path:str):
        """
        Object to manage the usage of the latent space variance. Uses Welfords algorithm from
        the OnlineVariance class, writes the results in a given file and plots the final latent
        usage heatmap over the epochs. Provides some functions to abstract the computations: 
        (LatentPlotter.update_latent_var(), LatentPlotter.reset(), LatentPlotter.plot_latent_usage())

        :param ndim: number of dimensions in the array
        :type ndim: int
        :param file_path: filename of the file where to store the computed variance at each epoch
        :type file_path: str
        """
        self.ndim = ndim
        self.file_path = file_path
        self.welfords_var = OnlineVariance(ndim=self.ndim, ddof=0)

        self.z_values = []
        self.var_history = []

    def __call__(self, z_values):
        self.update_latent_var(z_values)

    def update_latent_var(self, z_values):
        """
        Compute the online variance from the z sampled states of the latent space
        :param z_values: iterable of numpy.ndarray objects, list of samples from the latent space
        :type z_values: iterable (list)
        """
        [self.welfords_var.include(z_val) for z_val in z_values]
    
    def plot_latent_usage(self):
        """
        When called, uses the stored values of the variance for the different epochs to produce a 
        heatmap plot and saves the figure.
        """
        var_epochs = np.squeeze(np.array(self.var_history)).T
        plt.imshow(var_epochs, cmap='hot', interpolation='nearest')
        plt.xlabel("Epochs")
        plt.ylabel("Latent")
        plt.colorbar()
        plt.savefig("latent_usage.png")

    def reset(self):
        """
        To be used after finishig every epoch to reset the OnlineVariance object
        and so begin again to compute the variance for the new epoch.
        """
        self.var_history.append([self.welfords_var.std**2])

        with open(self.file_path, "a+") as f:
            print(self.var_history[-1][0], file=f)
        
        self.welfords_var = OnlineVariance(ndim=self.ndim, ddof=0)