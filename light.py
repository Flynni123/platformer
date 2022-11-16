from typing import *

import numpy as np
import pygame as pg

import maths
import shaders
import shapes
import settings


class lightColors:
    sun = (255, 224, 129)
    warm = (242, 229, 215)
    cold = (212, 229, 255)


class Light:

    def __init__(self, color=lightColors.warm, intensity=100.0):
        self.color = color
        self.color0 = (color[0], color[1], color[2], 0)
        self.intensity = intensity


class pointLight(Light):

    def __init__(self, xy, color=lightColors.warm, intensity=1.0, volume=0.3, spread=1.0):
        super().__init__(color, intensity)
        self.x, self.y = xy
        self.__volume = volume
        if 1 <= spread:
            self.__spread = spread
        else:
            raise ValueError(f"\'spread\' needs to be > 1")

    @property
    def pos(self): return self.x, self.y

    @property
    def volume(self): return self.__volume

    @property
    def spread(self): return self.__spread


class globalLight(Light):

    def __init__(self, color=lightColors.warm, intensity=1.0):
        super().__init__(color, intensity/20)


class LightHandler:
    def __init__(self, lights: List[Light], masks=None):
        """

        :param lights: list of lights in the scene
        :param masks: list of masks to be respected while rendering
        """
        if masks is None:
            masks = []
        self.lights = lights
        self.masks = masks

        self.shaderHandler = shaders.shaderHandler(settings.unscaledSize)

        self.composite = pg.Surface(settings.unscaledSize)
        self.globalLights = pg.Surface(settings.unscaledSize)
        self.g0 = np.zeros((settings.unscaledSize[1], settings.unscaledSize[0]),
                           dtype=np.float32)  # store Objects without light
        self.g1 = np.zeros((settings.unscaledSize[1], settings.unscaledSize[0]),
                           dtype=np.float32)  # store object normals
        self.mask = pg.Surface(settings.unscaledSize)  # store "layers" of shadow

        for m in self.masks:
            mTemp = m.to_surface(setcolor=(1, 1, 1), unsetcolor=(0, 0, 0))
            self.mask.blit(mTemp, (0, 0),
                           special_flags=pg.BLEND_RGBA_ADD)

        for l in self.lights:
            if isinstance(l, globalLight):
                lS = pg.Surface(settings.unscaledSize)
                lS.fill((round(l.color[0]*l.intensity), round(l.color[1]*l.intensity), round(l.color[2]*l.intensity)))
                self.globalLights.blit(lS, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

    def evaluate(self, g0: pg.Surface, g1: pg.Surface):
        self.composite.fill((0, 0, 0, 255))

        shaderInLights = []
        for l in self.lights:
            if isinstance(l, pointLight):
                shaderInLights.append(
                    [l.pos[0], l.pos[1], l.intensity, l.volume, l.color[0], l.color[1], l.color[2], l.spread]
                )

        shaderInLights = np.array(shaderInLights, dtype=np.float32)
        shaderInG0 = np.array(pg.surfarray.array2d(pg.transform.flip(pg.transform.rotate(g0, 90), False, True)),
                              dtype=np.float32)
        shaderInG1 = np.array(pg.surfarray.array2d(pg.transform.flip(pg.transform.rotate(g1, 90), False, True)),
                              dtype=np.float32)

        out = self.shaderHandler.fragment(shaderInLights, shaderInG0, shaderInG1)

        for y in range(settings.unscaledSize[1]):
            for x in range(settings.unscaledSize[0]):
                r, g, b = maths.getRGB(out[y][x])

                if settings.debug:
                    try:
                        self.composite.set_at((x, y), (r, g, b))
                    except ValueError:
                        print((r, g, b))
                        exit()
                elif settings.adjustColors:
                    r = 255 if r > 255 else r
                    g = 255 if g > 255 else g
                    b = 255 if b > 255 else b

                    r = 0 if r < 0 else r
                    g = 0 if g < 0 else g
                    b = 0 if b < 0 else b
                    self.composite.set_at((x, y), (r, g, b))
                else:
                    self.composite.set_at((x, y), (r, g, b))

        self.composite.blit(self.globalLights, (0, 0), special_flags=pg.BLEND_RGB_ADD)
        return self.composite


class test:
    clock = pg.time.Clock()
    pg.init()
    pg.display.init()

    def __init__(self):

        self.win = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.normals = pg.Surface(self.win.get_size())
        self.size = self.win.get_rect().size

        l = [pointLight((20, 20), (255, 255, 255)), pointLight((172, 88), (200, 0, 200))]

        lh = LightHandler(l)
        c = shapes.Circle(200, (255, 255, 255))

        self.run = True
        while self.run:

            self.win.fill((0, 0, 0))
            c.render(self.win, self.normals, pg.mouse.get_pos())
            cop = self.win.copy()
            self.win.fill((0, 0, 0))
            self.win.blit(pg.transform.scale(lh.evaluate(pg.transform.scale(cop, settings.unscaledSize),
                                                         pg.transform.scale(self.normals, settings.unscaledSize)), (1920, 1080)),
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
