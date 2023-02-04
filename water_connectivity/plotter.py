import argparse
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tqdm


def calculate_flux(h):
    """
    h has the form ts X ny X nx
    """
    boundary = h[:, 0:2, :]
    diff = np.diff(boundary, axis=1)
    flux = -np.sum(diff, axis=(1, 2))
    return flux


def create_animation_func(ts, data, pbar):
    flux = calculate_flux(data[:, 2, :, :])

    fig, axs = plt.subplots(1, 3)
    ax_b = axs[0]
    ax_h = axs[1]
    ax_f = axs[2]

    fig.tight_layout(h_pad=2.0, w_pad=5.0)

    ax_f.plot(ts, flux)
    ax_f.grid()

    t0 = ts[0]
    title = fig.suptitle(f"$t = {t0:.1f}[yr]$")
    b0, w0, h0 = data[0, :, :, :]

    b = data[0]
    h = data[2]

    b_lims = (np.min(b), np.max(b))
    im_b = ax_b.imshow(b0, cmap="YlGn", clim=b_lims, origin="lower")
    # title_b = ax_b.set_title("$b(x,y)$")
    fig.colorbar(im_b, fraction=0.046, pad=0.04)

    h_lims = (np.min(h), np.max(h))
    im_h = ax_h.imshow(h0, cmap="YlGn", clim=h_lims, origin="lower")
    fig.colorbar(im_h, fraction=0.046, pad=0.04)

    # fig.set_figheight(4)
    # fig.set_figwidth(12)

    def animation_func(frame):
        t = frame[0]
        pbar.update(1)
        b, _, h = frame[1]

        title.set_text(f"$t={t:.1f}$")
        im_b.set_array(b)
        im_h.set_array(h)

        plt.draw()
        return [im_b]

    return fig, zip(ts[1:], data[1:, :, :, :]), animation_func


def plot(path_raw, path_video):
    with h5py.File(path_raw, "r") as f:
        data = np.array(f["data"]).transpose(0, 1, 3, 2)
        ts = np.array(f["times"])

    pbar = tqdm.tqdm(total=len(ts))
    fig, frames, animation_func = create_animation_func(ts, data, pbar)
    ani = animation.FuncAnimation(
        fig,
        animation_func,
        frames,
        blit=True,
        interval=1,
        repeat=False,
        save_count=1000,
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
