from typing import *

import numpy as np
import pygame as pg

import maths
import light_shaders as shaders
import shapes
import settings
import colors


class Light:

    def __init__(self, color=colors.lightColors.cold, intensity=100.0):
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

    def reset(self):

        self.g0 = pg.Surface(self.size)
        self.g1 = pg.Surface(self.size)  # TODO: fix the weird thing where the g1 doesn´t get blit

    def prepareForShader(self):

        self.g0.convert()
        self.g1.convert()

        return np.array(pg.surfarray.array2d(self.g0), dtype=settings.dtype), \
               np.array(pg.surfarray.array2d(self.g1), dtype=settings.dtype)

    def blit(self, gBuffer, dest):
        assert isinstance(gBuffer, GBuffer)

        gBuffer.g0.convert()
        gBuffer.g1.convert()

        self.g0.blit(gBuffer.g0, dest)
        self.g1.blit(gBuffer.g1, dest)

    def upscale(self, fac=settings.scaleFactor):
        pos = (round(self.g0.get_size()[0] * fac), round(self.g0.get_size()[1] * fac))
        self.g0 = pg.transform.scale(self.g0, pos)
        self.g1 = pg.transform.scale(self.g1, pos)

    def downscale(self, fac=settings.scaleFactor):
        pos = (round(self.g0.get_size()[0] / fac), round(self.g0.get_size()[1] / fac))
        self.g0 = pg.transform.scale(self.g0, pos)
        self.g1 = pg.transform.scale(self.g1, pos)

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
        self.mask = pg.Surface(settings.unscaledSize)  # store "layers" of shadow

        for m in self.masks:
            mTemp = m.to_surface(setcolor=(1, 1, 1), unsetcolor=(0, 0, 0))
            self.mask.blit(mTemp, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

        for l in self.lights:
            if isinstance(l, globalLight):
                lS = pg.Surface(settings.unscaledSize)
                lS.fill(
                    (round(l.color[0] * l.intensity), round(l.color[1] * l.intensity), round(l.color[2] * l.intensity)))
                self.globalLights.blit(lS, (0, 0), special_flags=pg.BLEND_RGBA_ADD)

    def evaluate(self, win: GBuffer):
        self.composite.fill((0, 0, 0, 255))

        shaderInLights = []
        for l in self.lights:
            if isinstance(l, pointLight):

                shaderInLights.append(
                    [l.pos[0], l.pos[1], l.intensity, l.volume, l.color[0], l.color[1], l.color[2], l.spread]
                )

        shaderInLights = np.array(shaderInLights, dtype=np.float32)
        shaderInG0, shaderInG1 = win.prepareForShader()

        out = self.shaderHandler.fragment(shaderInLights, shaderInG0, shaderInG1)

        pg.pixelcopy.array_to_surface(self.composite, out.round(0).astype(np.uint32))
        self.composite.blit(self.globalLights, (0, 0), special_flags=pg.BLEND_RGB_ADD)

        return GBuffer(self.composite.get_size(), self.composite)


class test:
    clock = pg.time.Clock()
    pg.init()
    pg.display.init()

    def __init__(self):

        self.disp = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.win = GBuffer(self.disp.get_size())
        self.size = self.disp.get_rect().size

        l = [pointLight((20, 20), (255, 255, 255)), pointLight((172, 88), (200, 0, 200))]

        lh = LightHandler(l)
        c = shapes.Circle(200, (255, 255, 255))

        self.run = True
        while self.run:

            self.win.reset()
            c.render(self.win, pg.mouse.get_pos())
            cop = self.win.__copy__()
            cop.downscale()
            cop = lh.evaluate(cop)
            cop.upscale()
            self.win.reset()
            self.win.blit(cop, (0, 0))

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            self.disp.blit(self.win.g0, (0, 0))
            pg.display.update()
            self.clock.tick()
            x = self.clock.get_fps()
            if not self.run:
                print(round(x))


if __name__ == '__main__':
    test()
