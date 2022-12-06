import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def get_last_storage(storage_dir):
    files = sorted(os.listdir(storage_dir))
    return os.path.join(storage_dir, files[-1])


def create_animation_func(ts, data):
    fig, (ax_b, ax_w, ax_h) = plt.subplots(1, 3)

    t0 = ts[0]
    b0, w0, h0 = data[0, :, :, :]

    im_b = ax_b.imshow(b0, cmap="YlGn")
    fig.colorbar(im_b)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    im_w = ax_w.imshow(w0)
    im_h = ax_h.imshow(h0)

    def animation_func(frame):
        t = frame[0]
        print(t)
        b, w, h = frame[1]

        im_b.set_array(b)
        im_w.set_array(w)
        im_h.set_array(h)
        plt.draw()
        return [im_b, im_w, im_h]

    return fig, zip(ts[1:], data[1:, :, :, :]), animation_func


def main():
    storage_path = get_last_storage("data")
    with h5py.File(storage_path, "r") as f:
        data = np.array(f["data"])
        ts = np.array(f["times"])

    fig, frames, animation_func = create_animation_func(ts, data)
    ani = animation.FuncAnimation(
        fig, animation_func, frames, blit=True, interval=1, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
