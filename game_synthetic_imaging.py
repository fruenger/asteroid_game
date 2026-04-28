"""Procedural perfect-image stack for AsteroidGameState (no Ursina)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def build_synthetic_perfect_stack(*, imsize: int = 250, nstars: int = 100) -> tuple[list[np.ndarray], np.ndarray]:
    traj_angle = np.random.uniform(0, 2.0 * np.pi)
    image_locations = np.array([[np.cos(traj_angle), np.sin(traj_angle)], [-np.cos(traj_angle), -np.sin(traj_angle)]])
    image_locations = np.insert(image_locations, 1, np.mean(image_locations, axis=1), axis=1)
    image_locations *= np.random.uniform(4.0, 20.0)
    image_locations += np.expand_dims(np.random.uniform(imsize / 4.0, imsize * 3.0 / 4.0, 2), axis=1)
    image_locations = image_locations.astype(int)

    image_perfect_bg = np.ones((imsize, imsize), dtype=float) / 50.0
    image_perfect_bg[np.random.randint(0, imsize, nstars), np.random.randint(0, imsize, nstars)] = np.random.exponential(
        0.5, nstars
    )

    all_images_perfect: list[np.ndarray] = []
    for x, y in image_locations.T:
        image_perfect = image_perfect_bg.copy()
        image_perfect[x, y] = 1.0
        image_perfect = gaussian_filter(image_perfect, sigma=1.5)
        all_images_perfect.append(image_perfect)

    return all_images_perfect, image_locations
