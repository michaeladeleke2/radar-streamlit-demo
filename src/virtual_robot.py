from dataclasses import dataclass, field


@dataclass
class VirtualVexRobot:
    """
    Simple virtual robot:
    - 2D grid position (x, y)
    - 'up' increments y
    - 'right' increments x
    - 'push' increments push_count
    - Renders as an ASCII grid so movement is obvious in Streamlit.
    """
    x: int = 0
    y: int = 0
    push_count: int = 0
    history: list = field(default_factory=list)

    # Visual settings
    grid_w: int = 11   # columns
    grid_h: int = 11   # rows
    origin_center: bool = True  # show (0,0) near center of grid

    def reset(self):
        self.history.clear()
        self.x = 0
        self.y = 0
        self.push_count = 0

    def undo_last(self):
        if not self.history:
            return
        last = self.history.pop()
        kind = last["type"]
        if kind == "move":
            self.x -= last["dx"]
            self.y -= last["dy"]
        elif kind == "push":
            self.push_count = max(0, self.push_count - 1)

    def apply_gesture(self, label: str):
        """
        label expected: push, right, up (tolerant to variations)
        """
        lbl = (label or "").strip().lower()

        if lbl in ("up", "forward"):
            self._move(dx=0, dy=1, label="up")
        elif lbl == "right":
            self._move(dx=1, dy=0, label="right")
        elif lbl == "push":
            self._push()
        # ignore unknown labels

    def _move(self, dx: int, dy: int, label: str):
        self.x += dx
        self.y += dy
        self.history.append({"type": "move", "dx": dx, "dy": dy, "label": label})

    def _push(self):
        self.push_count += 1
        self.history.append({"type": "push", "label": "push"})

    def _grid_coords(self, x: int, y: int):
        """
        Convert world coords (x,y) into grid indices (gx, gy).
        gy=0 is top row for printing.
        """
        if self.origin_center:
            cx = self.grid_w // 2
            cy = self.grid_h // 2
            gx = cx + x
            gy = cy - y
        else:
            # origin at bottom-left (0,0)
            gx = x
            gy = (self.grid_h - 1) - y
        return gx, gy

    def _clamped(self):
        # Optional: keep robot inside the visible grid by clamping
        # (you can remove clamping if you want "infinite" coords)
        half_w = self.grid_w // 2
        half_h = self.grid_h // 2
        if self.origin_center:
            self.x = max(-half_w, min(half_w, self.x))
            self.y = max(-half_h, min(half_h, self.y))
        else:
            self.x = max(0, min(self.grid_w - 1, self.x))
            self.y = max(0, min(self.grid_h - 1, self.y))

    def render_text(self) -> str:
        """
        Display-friendly summary with an ASCII grid.
        """
        # Keep it visible
        self._clamped()

        # Build empty grid
        grid = [["·" for _ in range(self.grid_w)] for _ in range(self.grid_h)]

        # Mark origin
        ox, oy = self._grid_coords(0, 0)
        if 0 <= ox < self.grid_w and 0 <= oy < self.grid_h:
            grid[oy][ox] = "+"

        # Mark robot
        rx, ry = self._grid_coords(self.x, self.y)
        if 0 <= rx < self.grid_w and 0 <= ry < self.grid_h:
            grid[ry][rx] = "R"

        # Pretty print
        lines = []
        lines.append("Grid (R = robot, + = origin, · = empty)")
        for row in grid:
            lines.append(" ".join(row))

        last_action = self.history[-1]["label"] if self.history else "none"

        lines.append("")
        lines.append(f"Position (x, y): ({self.x}, {self.y})")
        lines.append(f"Last action: {last_action}")
        lines.append(f"Push actions: {self.push_count}")

        # Show last few history items
        tail = self.history[-8:]
        if tail:
            lines.append("Recent actions:")
            for item in tail:
                if item["type"] == "move":
                    lines.append(f"- move {item['label']} (dx={item['dx']}, dy={item['dy']})")
                else:
                    lines.append("- push")
        else:
            lines.append("Recent actions: (none)")

        return "\n".join(lines)