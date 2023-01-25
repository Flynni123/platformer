from typing import *

import numpy as np
import pygame as pg

import colors
import light_shaders as shaders
import settings


class Light:

    def __init__(self, color=colors.lightColors.cold, intensity=100.0):
        """

        :param color: obv
        :param intensity: 0-100 intensity
        """
        self.color = color
        self.color0 = (color[0], color[1], color[2], 0)
        self.intensity = intensity


class pointLight(Light):

    def __init__(self, xy, color=colors.lightColors.cold, intensity=1.0, volume=0.3, spread=1.0, parallax=1.0):
        super().__init__(color, intensity)
        self.x, self.y = xy
        self.__volume = volume
        self.parallax = parallax
        if 1 <= spread:
            self.__spread = spread
        else:
            raise ValueError(f"\'spread\' needs to be > 1")

    @property
    def pos(self):
        return self.x, self.y

    @property
    def volume(self):
        return self.__volume

    @property
    def spread(self):
        return self.__spread


class globalLight(Light):

    def __init__(self, color=colors.lightColors.cold, intensity=1.0):
        super().__init__(color, intensity / 20)


class GBuffer:
    """
    saves color info
    g0: visible layer
    g1: normal layer
    """

    def __init__(self, size=settings.unscaledSize, g0=None, g1=None, parallaxFac=1):
        self.parallaxFac = parallaxFac
        self.size = size

        if g0 is not None:
            self.g0 = g0  # pg.Surface(self.size)
            # self.g0.blit(g0, (0, 0))
        else:
            self.g0 = pg.Surface(self.size, pg.SRCALPHA)

        if g1 is not None:
            self.g1 = g1  # pg.Surface(self.size)
            # self.g1.blit(g1, (0, 0))
        else:
            self.g1 = pg.Surface(self.size, pg.SRCALPHA)

        self.g0shader = None
        self.g1shader = None

    def reset(self):

        self.g0 = pg.Surface(self.size)
        self.g1 = pg.Surface(self.size)

    def prepareForShader(self):

        self.g0.convert()
        self.g1.convert()

        self.g0shader = np.array(pg.surfarray.array2d(self.g0), dtype=settings.dtype)
        self.g1shader = np.array(pg.surfarray.array2d(self.g1), dtype=settings.dtype)

        return self.g0shader, self.g1shader

    def arrayToGBuffer(self, g0=None, g1=None):
        if g0 is not None:
            pg.surfarray.blit_array(self.g0, g0)

        if g1 is not None:
            pg.surfarray.blit_array(self.g1, g1)

    def blit(self, gBuffer, pos, ignoreG1=False):
        assert isinstance(gBuffer, GBuffer)

        gBuffer.g0.convert()
        gBuffer.g1.convert_alpha()

        self.g0.blit(gBuffer.g0, pos)
        if not ignoreG1:
            self.g1.blit(gBuffer.g1, pos, special_flags=pg.BLEND_RGBA_MAX)
            self.g1.convert()

    def upscale(self, fac=settings.scaleFactor):
        pos = (round(self.g0.get_size()[0] * fac), round(self.g0.get_size()[1] * fac))
        self.g0 = pg.transform.scale(self.g0, pos)
        self.g1 = pg.transform.scale(self.g1, pos)

    def downscale(self, fac=settings.scaleFactor):
        pos = (round(self.g0.get_size()[0] / fac), round(self.g0.get_size()[1] / fac))
        self.g0 = pg.transform.scale(self.g0, pos)
        self.g1 = pg.transform.scale(self.g1, pos)

    def flip(self):
        return GBuffer(self.size, pg.transform.flip(self.g0, True, False), pg.transform.flip(self.g1, True, False))

    def __copy__(self):
        g = GBuffer(self.size, self.g0, self.g1)
        return g

    def __getitem__(self, item):
        if item == 0:
            return self.g0
        elif item == 1:
            return self.g1
        else:
            raise ValueError


class LightHandler:
    def __init__(self, lights: List[Light]):
        """

        :param lights: list of lights in the scene
        """
        self.lights = lights

        self.shaderHandler = shaders.shaderHandler(settings.unscaledSize)

        self.composite = pg.Surface(settings.unscaledSize)
        self.globalLights = pg.Surface(settings.unscaledSize)

        self.shaderInLights = []
        self.lightsToList()

        for l in self.lights:
            if isinstance(l, globalLight):
                lS = pg.Surface(settings.unscaledSize)
                lS.fill(
                    (round(l.color[0] * l.intensity), round(l.color[1] * l.intensity), round(l.color[2] * l.intensity)))
                self.globalLights.blit(lS, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

    def evaluate(self, win: GBuffer):
        assert isinstance(self.shaderInLights, np.ndarray)
        assert len(self.shaderInLights) > 0
        shaderInG0, shaderInG1 = win.prepareForShader()
        self.lightsToList()

        out = self.shaderHandler.fragment(self.shaderInLights, shaderInG0, shaderInG1)

        pg.pixelcopy.array_to_surface(self.composite, out.round(0).astype(np.uint32))
        self.composite.blit(self.globalLights, (0, 0), special_flags=pg.BLEND_RGB_ADD)

        return GBuffer(self.composite.get_size(), self.composite)

    def lightsToList(self):
        del self.shaderInLights
        self.shaderInLights = []
        for l in self.lights:
            if isinstance(l, pointLight):
                self.shaderInLights.append(
                    [l.pos[0], l.pos[1], l.intensity, l.volume, l.color[0], l.color[1], l.color[2], l.spread, 0]
                )

        self.shaderInLights = np.array(self.shaderInLights, dtype=settings.dtype)
