import math


class Frame:
    def __init__(self, x, y, a, b, deep_h, deep_w=None):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.deep_h = deep_h
        if deep_w is None:
            self.deep_w = deep_h
        else:
            self.deep_w = deep_w

    def getcentre(self, x=None, y=None, a=None, b=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        # x /= self.deep_h
        y /= self.deep_h
        # a /= self.deep_h
        b /= self.deep_h
        if self.deep_w is not None:
            x /= self.deep_w
            # y /= self.deep_w
            a /= self.deep_w
            # b /= self.deep_w
        else:
            x /= self.deep_h
            a /= self.deep_h
        # if a-x>=b-y:
        #     h=(b-y)/2
        #     w=(x-a)/2
        #     c=math.sqrt(w*w-h*h)
        #     return [[(x+a)/2+c,(b+y)/2],[(x+a)/2-c,(b+y)/2]]
        # else:
        #     w = (b - y) / 2
        #     h = (x - a) / 2
        #     c = math.sqrt(w * w - h * h)
        #     return [[(x + a) / 2, (b + y) / 2+c], [(x + a) / 2, (b + y) / 2-c]]

        midx = (x + a) / 2
        midy = (y + b) / 2
        # return [[(x+midx)/2,(y+midy)/2],[(x+midx)/2,(b+midy)/2],[(a+midx)/2,(y+midy)/2],[(a+midx)/2,(b+midy)/2]]
        return [[midx, midy]]
