import pygame as pg
import numpy as np

import light
import shaders
import settings


class Camera:

    def __init__(self, shaderHandler: shaders.shaderHandler, exposure=1, blur=0):
        self._exposure = exposure
        self._blur = blur

        self.shaderHandler = shaderHandler
        self.out = pg.Surface(settings.unscaledSize)

    def preRender(self):

        attributesIn = np.array([self.exposure, self.blur], dtype=settings.dtype)

        self.shaderHandler.setAttributes(attr=attributesIn)

    def render(self):
        return self.shaderHandler.getResult()

    @property
    def blur(self): return self._blur

    @blur.setter
    def blur(self, new): self._blur = new

    @property
    def exposure(self): return self._exposure

    @exposure.setter
    def exposure(self, new): self._exposure = new

    @property
    def scaleExposure(self): return self._scaleExp

    @scaleExposure.setter
    def scaleExposure(self, new: bool): self._scaleExp = new
