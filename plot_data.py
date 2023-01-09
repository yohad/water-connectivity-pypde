import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def get_last_storage(storage_dir):
    files = sorted(os.listdir(storage_dir))
    return os.path.join(storage_dir, files[-1])


def create_animation_func(ts, data):
    fig, ax_b = plt.subplots(1, 1)

    t0 = ts[0]
    title = fig.suptitle(f"$t = {t0:.1f}$")
    b0, w0, h0 = data[0, :, :, :]

    lims = (0, 0.3)

    im_b = ax_b.imshow(b0, cmap="YlGn", clim=lims, origin="lower")
    title_b = ax_b.set_title(f"$b(x,y)$")
    fig.colorbar(im_b, fraction=0.046, pad=0.04)

    # fig.set_figheight(4)
    # fig.set_figwidth(12)

    def animation_func(frame):
        t = frame[0]
        print(f"{t:.2f}")
        b, w, h = frame[1]

        title.set_text(f"$t={t:.1f}$")
        im_b.set_array(b)

        plt.draw()
        return [im_b]

    return fig, zip(ts[1:], data[1:, :, :, :]), animation_func


def main():
    storage_path = get_last_storage("storage")
    # storage_path = "storage/storage-1207_144632.h5"
    with h5py.File(storage_path, "r") as f:
        data = np.array(f["data"]).transpose(0, 1, 3, 2)
        ts = np.array(f["times"])

    fig, frames, animation_func = create_animation_func(ts, data)
    ani = animation.FuncAnimation(
        fig, animation_func, frames, blit=True, interval=1, repeat=False, save_count=1000)

    # plt.show()
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save('animation.mp4', writer=FFwriter)


if __name__ == "__main__":
    main()
