import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Simulation(object):
    def __init__(self):
        self._frames = []

    def append(self, frame):
        """
        Add data to the simulation
        :param frame: 2D matrix, all frames of a single simulation must have the same size
        """
        self._frames.append(frame)

    def visualize(self, path):
        """
        Saves a GIF file of the simulation
        :param path: Path to save the file to
        """
        fig, ax = plt.subplots()

        frames = []
        for i, frame in enumerate(self._frames):

            if i == 0:
                current_frame = ax.imshow(frame, vmin=0, vmax=1)
            else:
                current_frame = ax.imshow(frame, vmin=0, vmax=1, animated=True)

            frames.append([current_frame])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        output_path = os.path.join(path, f"{time.strftime('%H_%M_%S')}.gif")
        ani.save(output_path)
