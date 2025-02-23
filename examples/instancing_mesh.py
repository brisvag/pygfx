"""
Example rendering the same mesh object multiple times, using instancing.
"""

import numpy as np
import imageio
import pygfx as gfx

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:bricks.jpg").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.TorusKnotGeometry(1, 0.3, 128, 32)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshPhongMaterial(map=tex)
obj = gfx.InstancedMesh(geometry, material, 100)
scene.add(obj)


# Set matrices. Note that these are sub-transforms of the mesh's own matrix.
for y in range(10):
    for x in range(10):
        m = gfx.linalg.Matrix4().set_position_xyz(y * 2, x * 2, 0)
        obj.set_matrix_at(x + y * 10, m.elements)


camera = gfx.PerspectiveCamera(70, 1)
camera.position.set(9, 9, 15)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec()
