import gmpy2 as gmpy
import numpy as np
from datetime import datetime

class Trace:
    def __init__(self):
        pass

class EdgeTrace(Trace):
    def __init__(self, data, edge):
        #self._trace = [int.from_bytes(sample, 'little') for sample in data]
        
        self._trace = [int.from_bytes(sample, 'big') for sample in data]
        self.__edge = edge # True = Positive Edge, False = Negative Edge

    @property
    def edge(self):
        return self._edge

    @property
    def pop(self):
        return np.array([gmpy.popcount(sample) for sample in self._trace])

    def __str__(self):
        s = ''
        for d in self._trace:
            s = s + ('{:0256b}'.format(d))[::-1] +'\n'
        return s[:-1]

class PositiveTrace(EdgeTrace):
    def __init__(self, data):
        super().__init__(data, True)

    @property
    def first(self):
        return np.array([gmpy.bit_scan0(sample) for sample in self._trace])

    @property
    def last(self):
        return np.array([gmpy.bit_length(sample) for sample in self._trace])

class NegativeTrace(EdgeTrace):
    def __init__(self, data):
        super().__init__(data, False)

    @property
    def first(self):
        return np.array([gmpy.bit_scan1(sample) for sample in self._trace])

    @property
    def last(self):
        return np.array([gmpy.bit_length(sample) for sample in self._trace])

class CombinedTrace(Trace):
    def __init__(self, ts):
        poss = ts[1::4]
        negs = ts[3::4]

        self.__pos = PositiveTrace(poss)
        self.__neg = NegativeTrace(negs)
        super().__init__()

    @property
    def pos(self):
        return self.__pos

    @property
    def neg(self):
        return self.__neg
