"""AABB rect nhẹ, không phụ thuộc `pygame`, để game core chạy headless thuần
Python (nguyên tắc §3.1: core không gọi `pygame.display`).

Ngữ nghĩa khớp `pygame.Rect`: `right = x + width`, `bottom = y + height`,
`colliderect` không tính hai cạnh chỉ chạm nhau là va chạm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def centerx(self) -> float:
        return self.x + self.width / 2

    @property
    def centery(self) -> float:
        return self.y + self.height / 2

    def colliderect(self, other: Rect) -> bool:
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )
