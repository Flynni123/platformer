import pygame as pg

import colors
import light
import scene

"""
Instead of rendering the object and passing it to the Display class, i'm passing the object to render to the Display
class and then Rendering it. if i want to change it, i just change it.
- Tzu Sun, art of not knowing what i'm programming
"""


class pixelatingMethods:
    lame = 0
    fancy = 1


class Display:

    def __init__(self, size=(0, 0), fullscreen=True, time=scene.TimeHandler()):
        """
        :param size: window size
        :param fullscreen: if window should be in fullscreen. ignores size argument
        :param time: handles time
        """

        self.size = size
        self.fullscreen = fullscreen
        self.timeHandler = time

        # init EVERYTHING
        pg.init()
        pg.display.init()
        pg.mixer.init()

        self.clock = pg.time.Clock()

        if fullscreen:
            self.disp = pg.display.set_mode(size, pg.FULLSCREEN, pg.SRCALPHA)
        else:
            self.disp = pg.display.set_mode(size, pg.SRCALPHA)

        self.s = scene.Scene(scene.SceneLayout(scene.ImageHandler([
            "assets/images/bg2.png",
            "assets/images/bg1.png",
            "assets/images/bg0.png"
        ]), light.LightHandler([
            light.pointLight((10, 10), light.lightColors.sun, 1, .5, 1),
            light.globalLight(intensity=.05)
        ])), scene.Character(pg.Surface((0, 0))), True)

        self.run = False
        firstRun = True

        self.run = True
        while self.run:

            if firstRun:
                firstRun = False
                self.timeHandler.start()

            self.disp.fill(colors.background)

            ticks = self.timeHandler.getTicks()
            self.s.update(ticks, pg.key.get_pressed())

            for event in pg.event.get():

                if event.type == pg.QUIT:

                    self.run = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_ESCAPE:
                        self.run = False

            self.disp.blit(self.s.render(), (0, 0))
            pg.display.update()
            self.clock.tick()
            if not self.run:
                print(round(self.clock.get_fps(), 1))


d = Display()
