"""
Display line segments. Can be useful e.g. for visializing vector fields.
"""

import numpy as np

import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()

x = np.linspace(20, 620, 200, dtype=np.float32)
y = np.sin(x / 10) * 100 + 200

positions = np.column_stack([x, y, np.zeros_like(x)])
colors = np.random.uniform(0, 1, (x.size, 4)).astype(np.float32)
geometry = gfx.Geometry(positions=positions, colors=colors)


# Also see LineSegmentMaterial and LineThinSegmentMaterial
material = gfx.LineSegmentMaterial(
    thickness=6.0, color=(0.0, 0.7, 0.3, 0.5), vertex_colors=True
)
line = gfx.Line(geometry, material)
scene.add(line)

camera = gfx.ScreenCoordsCamera()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
