import math
import numpy as np


def smoothStep(edge0, edge1, x):
    x = np.clip((x - edge0)/(edge1 - edge0), 0.0, 1.0)
    return x*x*(3 - 2*x)


def distance(pos1, pos2):
    a_ = pos1[0] - pos2[0]
    b_ = pos1[1] - pos2[1]

    return abs(pow(pow(a_, 2) + pow(b_, 2), .5))


def fromPolar(r, roh):
    degRoh = roh * (math.pi / 180)
    return r * math.sin(degRoh), - (r * math.cos(degRoh))


class Vec2:
    def __init__(self, xy, a=0.0):
        self.x, self.y = xy
        self._a = a

    def __sub__(self, other):
        return Vec2((self.x - other.x, self.y - other.y), self.a)

    def __add__(self, other):
        return Vec2((self.x + other.x, self.y + other.y), self.a)

    def __mul__(self, other):
        return Vec2((self.x * other.x, self.y * other.y), self.a)

    def __truediv__(self, other):
        return Vec2((self.x / other.x, self.y / other.y), self.a)

    def __abs__(self):
        return Vec2((abs(self.x), abs(self.y)), abs(self._a))

    def __getitem__(self, item):
        if item < 2:
            return self.x if item == 0 else self.y
        else:
            raise KeyError

    def copy(self):
        return Vec2((self.x, self.y), self._a)

    @property
    def a(self):
        return round(self._a)

    @property
    def pos(self):
        return round(self.x), round(self.y)

    @a.setter
    def a(self, value):
        self._a = value
        if self._a < -180:
            self._a += 360
        elif self._a > 180:
            self._a -= 360

    def rotate(self, a):
        self.a += a

    def move(self, dst):
        rad = (self.a * math.pi) / 180
        rot = np.array([[math.cos(rad), -math.sin(rad)],
                        [math.sin(rad), math.cos(rad)]])

        dot = np.dot(rot, [0, dst])

        self.x -= float(dot.T[0])
        self.y -= float(dot.T[1])

    def __len__(self):
        return distance((0, 0), self.pos)

    def normalize(self):
        l = self.__len__()
        self.x /= l
        self.y /= l

    def fromPolar(self, r, roh):
        degRoh = roh * (math.pi / 180)
        self.x = r * math.sin(degRoh)
        self.y = - (r * math.cos(degRoh))


class Vec3:
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3((self.x - other, self.y - other, self.z - other))
        else:
            return Vec3((self.x - other.x, self.y - other.y, self.z - other.z))

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3((self.x + other, self.y + other, self.z + other))
        else:
            return Vec3((self.x + other.x, self.y + other.y, self.z + other.z))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3((self.x * other, self.y * other, self.z * other))
        else:
            return Vec3((self.x * other.x, self.y * other.y, self.z * other.z))

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3((self.x / other, self.y / other, self.z / other))
        else:
            return Vec3((self.x / other.x, self.y / other.y, self.z / other.z))

    def __abs__(self):
        return Vec3((abs(self.x), abs(self.y), abs(self.z)))

    def copy(self):
        return Vec3((self.x, self.y, self.z))

    @property
    def pos(self):
        return round(self.x), round(self.y), round(self.z)
