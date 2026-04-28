"""On-screen typed message and small UI helpers."""

from __future__ import annotations

import time as time_stdlib

import numpy as np
from ursina import Text


def blink_opacity(duration: float, min_alpha: float = 0.25) -> float:
    return min_alpha + (np.sin(np.pi * time_stdlib.time() / duration) ** 2) * (1.0 - min_alpha)


class OnScreenMessage(Text):
    def __init__(self, message, time_between_letters=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_between_letters = time_between_letters
        self.message_triggered = time_stdlib.time()
        self.message = message
        self.text = message

    def write(self) -> None:
        finalliteral = int((time_stdlib.time() - self.message_triggered) / self.time_between_letters)
        finalliteral = int(np.clip(finalliteral, 0, len(self.message)))
        self.text = self.message[:finalliteral]

    def reset_timer(self) -> None:
        self.message_triggered = time_stdlib.time()
