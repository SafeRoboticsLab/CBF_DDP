from typing import Optional
import numpy as np

from matplotlib import pyplot as plt


class CircleFootprint:

    def __init__(
        self, center: Optional[np.ndarray] = None,
        ego_radius: float = 0
    ) -> None:
        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = center.copy()
        self.ego_radius = ego_radius

    def set_center(self, center: Optional[np.ndarray] = None):
        self.center = center

    def plot(self, ax, color='r', lw=1.5, alpha=1.):
        ego_circle = plt.Circle(
            [self.center[0], self.center[1]], self.ego_radius, alpha=0.4, color=color)
        ax.add_patch(ego_circle)
