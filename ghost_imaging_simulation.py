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
def random_mask(shape: tuple[int, int]) -> np.array:
    return np.random.random(shape)


@numba.njit
def overlay(sample: np.array, mask: np.array) -> np.array:
    return sample * mask


# @numba.njit
def ghost_image(sample: np.array, n: int, liveplot = False) -> np.array:
    """
    take n intensity measurements of the sample together with a random mask
    and return the ghost image
    """
    shape = sample.shape
    n_pixels = shape[0] * shape[1]
    intensity_expectation = overlay(sample, 0.5*np.ones(shape)).sum()  # shape + uniformly gray mask
    # print(f"{intensity_expectation=}")
    result = np.zeros(shape)
    if liveplot:
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(sample, cmap="Greys_r")
    for i in range(n):
        mask = random_mask(shape)
        intensity = overlay(sample, mask).sum()
        # print(f"{intensity=}")
        # add the mask with the intensity to the result
        result += (intensity - intensity_expectation) * mask
        if liveplot and i % 1000 == 0:
            ax[1].clear()
            ax[1].imshow(result / (i+1), cmap="Greys_r")
            ax[1].set_title(i)
            plt.pause(0.001)
    if liveplot:
        plt.show()
    return result / n


if __name__ == "__main__":
    sample = get_sample_image("input.bmp")
    result = ghost_image(sample, 500000, liveplot=True)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im1 = axes[0].imshow(sample, cmap="Greys_r")
    im2 = axes[1].imshow(result, cmap="Greys_r")
    # fig.colorbar(im2)
    # scale result to values 0 - 255
    lowest = np.min(result)
    highest = np.max(result)
    # print(lowest, highest)
    result = 255 / (highest - lowest) * (result - lowest)
    # plt.imsave("result.bmp", result, cmap="Greys_r")
    # plt.show()
