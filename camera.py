import pygame as pg
import numpy as np

import light
import camera_shaders
import settings


class Camera:

    def __init__(self, exposure=1, blur=False):
        self._exposure = exposure
        self._blur = blur

        self.shaderHandler = camera_shaders.shaderHandler(settings.unscaledSize)
        self.out = pg.Surface(settings.unscaledSize)

    def render(self, surf: light.GBuffer):

        imageIn = np.array(pg.surfarray.array2d(surf.g0), dtype=settings.dtype)
        attributesIn = np.array([self.exposure, 1 if self.blur else 0], dtype=settings.dtype)

        out: np.ndarray = self.shaderHandler.fragment(imageIn, attributesIn)
        pg.pixelcopy.array_to_surface(self.out, out.round(0).astype(np.uint32))

        return self.out

    @property
    def blur(self): return self._blur

    @blur.setter
    def blur(self, new: bool): self.blur = new

    @property
    def exposure(self): return self._exposure

    @exposure.setter
    def exposure(self, new): self._exposure = new
