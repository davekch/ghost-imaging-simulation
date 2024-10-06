import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import numba
import argparse


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
    ax[0].set_title("true image")
    for i in range(n):
        result += ghost_image_step(sample, intensity_expectation)
        if i % (n // min(n, 500)) == 0 or i == n - 1:
            ax[1].clear()
            ax[1].imshow(result / (i+1), cmap="Greys_r")
            ax[1].set_title(f"reconstructed image (n={i+1})")
            plt.pause(0.0001)
    plt.savefig("result.png")
    plt.show()
    return result / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-n", help="number of masks", default=50_000)
    parser.add_argument("--fast", help="no live plotting", action="store_true", default=False)
    parser.add_argument("--out", help="filename for reconstructed image", default="reconstructed.bmp")
    args = parser.parse_args()

    sample = get_sample_image(args.input)
    n = int(args.n)
    if args.fast:
        result = ghost_image(sample, n)
    else:
        result = ghost_image_liveplot(sample, n)
    plt.imsave(args.out, result, cmap="Greys_r")
