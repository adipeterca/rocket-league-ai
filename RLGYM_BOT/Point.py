class Point2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getPoint2D(self):
        return tuple(self.x, self.y)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val
    
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val
