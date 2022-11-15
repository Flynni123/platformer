from typing import *

import pygame as pg

import numpy as np

import gradients
import maths
import shaders


class settings:
    unscaledSize = (192, 108)


class lightColors:
    sun = (255, 224, 129, 255)
    warm = (242, 229, 215, 255)
    cold = (212, 229, 255, 255)


class Light:

    def __init__(self, color=lightColors.warm, intensity=100):
        self.color = color
        self.color0 = (color[0], color[1], color[2], 0)
        self.intensity = intensity


class pointLight(Light):

    def __init__(self, xy, color=lightColors.warm, intensity=1.0, volume=0.3):
        super().__init__(color, intensity)
        self.__xy = xy
        self.__volume = volume

    @property
    def pos(self): return self.__xy

    @property
    def volume(self): return self.__volume


class globalLight(Light):

    def __init__(self, color=lightColors.warm, intensity=1.0):
        super().__init__(color, intensity)


class LightHandler:
    def __init__(self, g0: pg.Surface, lights: List[Light], masks=None):
        """

        :param g0: part of G-Buffer which stores the un-shaded objects
        :param lights: list of lights in the scene
        :param masks: list of masks to be respected while rendering
        """
        if masks is None:
            masks = []
        self.g0IN = g0
        self.lights = lights
        self.masks = masks

        # TODO: add g1 (normals)
        self.composite = pg.Surface(settings.unscaledSize, pg.SRCALPHA)
        self.g0 = pg.Surface(settings.unscaledSize, pg.SRCALPHA)  # store Objects without light
        self.g1 = pg.Surface(settings.unscaledSize, pg.SRCALPHA)  # store object normals
        self.mask = pg.Surface(settings.unscaledSize)  # store "layers" of shadow
        self.light = pg.Surface(settings.unscaledSize, pg.SRCALPHA)
        self.lightVolume = pg.Surface(settings.unscaledSize, pg.SRCALPHA)

        lightBufferTemp = [pg.Surface(settings.unscaledSize, pg.SRCALPHA) for _ in range(len(self.lights))]
        self.lightBuffer = []
        for i in range(len(self.lights)):
            self.lightBuffer.append([lightBufferTemp[i], self.lights[i]])

        g0Mask = pg.mask.from_surface(self.g0IN, 1)
        self.masks.append(g0Mask)
        for m in self.masks:
            mTemp = m.to_surface(setcolor=(1, 1, 1), unsetcolor=(0, 0, 0))
            self.mask.blit(mTemp, (0, 0),
                           special_flags=pg.BLEND_RGBA_ADD)

        self.g0IN.convert(self.g0)
        self.g0.blit(self.g0IN, (0, 0))
        self.g0.set_colorkey((0, 0, 0))

    def evaluate(self, g0: pg.Surface):
        self.composite.fill((0, 0, 0, 255))
        self.composite.fill((0, 0, 0, 0))
        self.light.fill((0, 0, 0, 255))
        self.light.fill((0, 0, 0, 0))
        self.lightVolume.fill((0, 0, 0, 255))
        self.lightVolume.fill((0, 0, 0, 0))

        masksTemp = self.mask.copy()
        g0Mask = pg.mask.from_surface(g0, 1)
        mTemp = g0Mask.to_surface(setcolor=(1, 1, 1), unsetcolor=(0, 0, 0))
        masksTemp.blit(mTemp, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

        lSurfTemp = []
        lSurfVolTemp = []
        self.g0 = g0

        """
        turnaroundSize = settings.unscaledSize[1], settings.unscaledSize[0]

        comp = np.zeros(turnaroundSize, dtype=np.float64)
        shaderInLights = []
        shaderInLightSurf = []
        lSurfTemp = []
        for lSurf, l in self.lightBuffer:
            if isinstance(l, pointLight):
                shaderInLights.append(
                    [l.pos[0], l.pos[1], l.intensity, l.volume, l.color[0], l.color[1], l.color[2], l.color[3]]
                )
                shaderInLightSurf.append(np.zeros(turnaroundSize))
                lSurfTemp.append(pg.Surface(settings.unscaledSize, pg.SRCALPHA))

        shaderInLights = np.array(shaderInLights, dtype=np.float64)
        shaderInLightSurf = np.array(shaderInLightSurf, dtype=np.float64)

        shaders.fragment[settings.unscaledSize[1], settings.unscaledSize[0]](shaderInLights, shaderInLightSurf, comp)

        for i in range(len(lSurfTemp)):
            lSurfFrag = shaderInLightSurf[i]
            lSurf = lSurfTemp[i]
            l: pointLight = self.lights[i]
            lSurf.lock()
            for y in range(settings.unscaledSize[1] - 1):
                for x in range(settings.unscaledSize[0] - 1):
                    c = pg.Color(int(lSurfFrag[y][x]))
                    lSurf.set_at((x, y), c)
            lSurf.unlock()
            self.composite.blit(lSurf, (0, 0), special_flags=pg.BLEND_RGBA_ADD)
            if l.volume > 0:
                vSurf = pg.Surface(settings.unscaledSize)
                c = round(255 * l.volume)
                vSurf.fill((c, c, c))
                vSurf.blit(lSurf, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
                self.lightVolume.blit(vSurf, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

        self.composite.blit(g0, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        self.composite.blit(self.lightVolume, (0, 0), special_flags=pg.BLEND_RGBA_ADD)
        """

        turnaroundSize = settings.unscaledSize[1], settings.unscaledSize[0]
        comp = np.zeros(turnaroundSize, dtype=np.float32)
        shaderInLights = []
        shaderInLightSurf = []
        lSurfTemp = []
        for lSurf, l in self.lightBuffer:
            if isinstance(l, pointLight):
                shaderInLights.append(
                    [l.pos[0], l.pos[1], l.intensity, l.volume, l.color[0], l.color[1], l.color[2], l.color[3]]
                )
                shaderInLightSurf.append(np.zeros(turnaroundSize))
                lSurfTemp.append(pg.Surface(settings.unscaledSize, pg.SRCALPHA))

        shaderInLights = np.array(shaderInLights, dtype=np.float32)
        shaderInLightSurf = np.array(shaderInLightSurf, dtype=np.float32)

        shaders.fragment(settings.unscaledSize[0], settings.unscaledSize[1], shaderInLights, shaderInLightSurf, comp, comp)

        for i in range(len(lSurfTemp)):
            lSurfFrag = shaderInLightSurf[i]
            lSurf = lSurfTemp[i]
            l: pointLight = self.lights[i]
            lSurf.lock()
            for y in range(settings.unscaledSize[1] - 1):
                for x in range(settings.unscaledSize[0] - 1):
                    c = pg.Color(int(lSurfFrag[y][x]))
                    lSurf.set_at((x, y), c)
            lSurf.unlock()
            self.composite.blit(lSurf, (0, 0), special_flags=pg.BLEND_RGBA_ADD)
            if l.volume > 0:
                vSurf = pg.Surface(settings.unscaledSize)
                c = round(255 * l.volume)
                vSurf.fill((c, c, c))
                vSurf.blit(lSurf, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
                self.lightVolume.blit(vSurf, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

        self.composite.blit(g0, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        self.composite.blit(self.lightVolume, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

        return self.composite


class test:
    clock = pg.time.Clock()
    pg.init()
    pg.display.init()

    def __init__(self):

        self.win = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.size = self.win.get_rect().size

        l = [pointLight((20, 20), (200, 200, 100, 255)), pointLight((172, 88), (200, 0, 200, 255))]

        lh = LightHandler(self.win, l)

        self.run = True
        while self.run:

            self.win.fill((0, 0, 0, 255))
            pg.draw.circle(self.win, (200, 200, 200), pg.mouse.get_pos(), 200)
            self.win.blit(
                pg.transform.scale(lh.evaluate(pg.transform.scale(self.win, settings.unscaledSize)), (1920, 1080)),
                (0, 0))

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            pg.display.update()
            self.clock.tick()
            x = self.clock.get_fps()
            if not self.run:
                print(round(x))


if __name__ == '__main__':
    test()
