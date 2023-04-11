import argparse
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tqdm

from post_processing import get_water_flux, post_processing


def dict_to_frames(processed):
    ts = processed["times"]
    b = processed["vegetation"]
    flux = processed["flux"]

    frames = []
    for i in range(len(ts)):
        frames.append((ts[i], b[i], flux[i]))

    return frames


def create_animation_func(processed, pbar):
    ts = processed["times"]
    b = processed["vegetation"]
    flux = processed["flux"]
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.tight_layout(h_pad=2.0, w_pad=5.0)
    title = fig.suptitle(f"$t = {ts[0]:.1f}[yr]$")
    
    ax_b = axs[0]
    ax_b.set_title("Vegetation")
    b0 = b[0]
    b_lims = (0, np.max(b[20:]))
    im_b = ax_b.imshow(b0, cmap="YlGn", clim=b_lims, origin="lower")
    fig.colorbar(im_b, fraction=0.046, pad=0.04)
    
    ax_flux = axs[1]
    ax_flux.set_title("Water Flux")
    flux0 = flux[0]
    flux_lims = (np.min(flux), np.max(flux))
    im_flux, = ax_flux.plot(flux0)

    def animation_func(frame):
        pbar.update(1)
        t, b, flux = frame

        title.set_text(f"$t={t:.1f}[yr]$")
        
        im_b.set_array(b)
        im_flux.set_ydata(flux)

        plt.draw()
        return [im_b, im_flux]

    return fig, dict_to_frames(processed), animation_func


def plot(path_raw, path_video):
    processed = post_processing(path_raw)

    pbar = tqdm.tqdm(total=len(processed["times"]))
    fig, frames, animation_func = create_animation_func(processed, pbar)
    ani = animation.FuncAnimation(
        fig,
        animation_func,
        frames,
        blit=True,
        interval=1,
        repeat=False,
    )

    # plt.show()
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save(path_video, writer=FFwriter)
    pbar.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="input",
        help="Path to input h5 file",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="output",
        help="Path to output mp4 file",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    plot(args.input, args.output)


if __name__ == "__main__":
    main()
