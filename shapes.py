import pygame as pg

import maths


# r channel - 128: normal x
# g channel - 128: normal y


class Circle:
    def __init__(self, radius, color):
        self.__g0 = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)
        self.__g1 = pg.Surface((radius * 2, radius * 2), pg.SRCALPHA)

        self.rect = self.__g0.get_rect()
        self.center = self.rect.center

        self.radius = radius
        self.color = color

        self.__renderG0()
        self.__renderG1()

    def __renderG0(self):
        pg.draw.circle(self.__g0, self.color, self.center, self.radius)

    def __renderG1(self):

        for y in range(self.radius*2):
            for x in range(self.radius*2):
                if self.__g0.get_at((x, y)) == self.color:
                    self.__g1.set_at((x, y), (128+(x-self.center[0])/2, 128+(y-self.center[1])/2, 0))

    @property
    def g0(self): return self.__g0

    @property
    def g1(self): return self.__g1

    def render(self, gBuffer, dest):
        gBuffer[0].blit(self.__g0, (dest[0] - self.center[0], dest[1] - self.center[1]))
        gBuffer[1].blit(self.__g1, (dest[0] - self.center[0], dest[1] - self.center[1]))
