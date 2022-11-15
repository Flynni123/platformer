import pygame as pg


def color(r=None, g=0, b=0, a=255):
    if type(r) == int or type(r) == float:
        return pg.Color(r, g, b, a)
    elif type(r) == list or type(r) == tuple:
        return pg.Color(r)
    else:
        raise TypeError(f"invalid value of r: {r}")


black = color(0, 0, 0)
white = color(255, 255, 255)

background = color(150, 199, 211)
