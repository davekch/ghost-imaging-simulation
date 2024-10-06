import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import numba


def get_sample_image(path) -> np.array:
    img = image.imread(path)
    # if the image is rgb, make it monochromatic
    if len(img.shape) == 3:
        return img.mean(axis=2)
    return img / 255


@numba.njit
def ghost_image_step(sample: np.ndarray, intensity_expectation: float) -> np.ndarray:
    """
    reconstruct the image after *one* intensity measurement
    """
    mask = np.random.random(sample.shape)
    intensity = (sample * mask).sum()
    return (intensity - intensity_expectation) * mask


@numba.njit
def ghost_image(sample: np.ndarray, n: int) -> np.ndarray:
    """
    reconstruct the image after n intensity measurements
    """
    # the expectation value for the measured intensity is the sample + a uniformly gray mask
    intensity_expectation = (0.5 * sample).sum()
    result = np.zeros(sample.shape)
    for _ in range(n):
        result += ghost_image_step(sample, intensity_expectation)
    return result / n


def ghost_image_liveplot(sample: np.ndarray, n: int) -> np.array:
    """
    same as `ghost_image` but also create a live plot of the evolving picture
    """
    intensity_expectation = (0.5 * sample).sum()
    result = np.zeros(sample.shape)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(sample, cmap="Greys_r")
    for i in range(n):
        result += ghost_image_step(sample, intensity_expectation)
        if i % (n // 500) == 0 or i == n - 1:
            ax[1].clear()
            ax[1].imshow(result / (i+1), cmap="Greys_r")
            ax[1].set_title(i)
            plt.pause(0)
    plt.show()
    return result / n


if __name__ == "__main__":
    sample = get_sample_image("input.bmp")
    n = 5_000
    # result = ghost_image(sample, n)
    result = ghost_image_liveplot(sample, n)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im1 = axes[0].imshow(sample, cmap="Greys_r")
    im2 = axes[1].imshow(result, cmap="Greys_r")
    plt.imsave("reconstructed.bmp", result, cmap="Greys_r")
    plt.savefig("result.png")
    plt.show()
