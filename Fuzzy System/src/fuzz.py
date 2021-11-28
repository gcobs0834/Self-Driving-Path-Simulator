from enum import Enum


class Level(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


class Fuzz():
    def __init__(self):
        pass


class Fuzzifier():
    @staticmethod
    def to_level(s, m, l):
        return Level([s, m, l].index(max([s, m, l])))

    @staticmethod
    def l_point(input_):
        f = Fuzzifier()
        s = f.side_small(input_)
        m = f.side_medium(input_)
        l = f.side_large(input_)
        return Fuzzifier.to_level(s, m, l)

    @staticmethod
    def r_point(input_):
        f = Fuzzifier()
        s = f.side_small(input_)
        m = f.side_medium(input_)
        l = f.side_large(input_)
        return Fuzzifier.to_level(s, m, l)

    @staticmethod
    def c_point(input_):
        f = Fuzzifier()
        s = f.front_small(input_)
        m = f.front_medium(input_)
        l = f.front_large(input_)
        return Fuzzifier.to_level(s, m, l)

    def side_small(self, input_):
        if input_ < 5:
            return 1
        elif input_ < 7:
            return (7 - input_)/2
        else:
            return 0

    def side_medium(self, input_):
        if 4 < input_ and input_ <= 8:
            return (input_-4)/4
        elif 8 < input_ and input_ <= 12:
            return (12-input_)/4
        else:
            return 0

    def side_large(self, input_):
        if 9 < input_ and input_ <= 16:
            return (input_-9)/7
        elif input_ > 16:
            return 1
        else:
            return 0

    def front_small(self, input_):
        if input_ < 5:
            return 1
        elif input_ < 10:
            return (10 - input_)/5
        else:
            return 0

    def front_medium(self, input_):
        if 14 < input_ and input_ <= 16:
            return (input_-14)/2
        elif 16 < input_ and input_ <= 18:
            return (18-input_)/2
        else:
            return 0

    def front_large(self, input_):
        if input_ > 30:
            return 1
        else:
            return 0


class Rules():
    def __init__(self):
        pass

    @staticmethod
    def apply(l_point, c_point, r_point):
        if r_point == Level.SMALL:
            return -45
        if l_point == Level.SMALL:
            return 45
        if r_point == Level.MEDIUM and c_point == Level.SMALL:
            return -20
        if l_point == Level.MEDIUM and c_point == Level.SMALL:
            return 20
        return 0