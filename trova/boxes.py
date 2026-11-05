"""Axis-aligned boxes in pixel coordinates, plus the small helpers everything else shares."""

from dataclasses import dataclass, field


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    id: int = 0
    source: str = "blob"      # blob | grok | manual
    label: str = ""
    score: float = 1.0
    touches_border: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def w(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def h(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def coverage_by(self, other: "Box") -> float:
        """Fraction of this box that lies inside `other`."""
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        return (ix * iy) / self.area if self.area > 0 else 0.0

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1),
                   source=self.source, score=self.score + other.score,
                   touches_border=self.touches_border or other.touches_border, extra={**other.extra, **self.extra})

    def to_list(self) -> list[int]:
        return [int(round(v)) for v in (self.x0, self.y0, self.x1, self.y1)]

    def as_dict(self) -> dict:
        return {"id": self.id, "box": self.to_list(), "source": self.source,
                "label": self.label, "center": [round(c, 1) for c in self.center]}

    @classmethod
    def from_list(cls, xs, **kw) -> "Box":
        return cls(float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]), **kw)

    @classmethod
    def from_norm1000(cls, xs, width: int, height: int, **kw) -> "Box":
        x0, y0, x1, y1 = (float(v) for v in xs)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return cls(x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000, **kw)


def number_boxes(boxes: list[Box]) -> list[Box]:
    """Sort into reading order (rows top to bottom, then left to right) and number from 1."""
    if not boxes:
        return boxes
    avg_h = sum(b.h for b in boxes) / len(boxes)
    row = max(avg_h * 0.6, 1.0)
    boxes = sorted(boxes, key=lambda b: (round(b.center[1] / row), b.center[0]))
    for i, b in enumerate(boxes, start=1):
        b.id = i
    return boxes
