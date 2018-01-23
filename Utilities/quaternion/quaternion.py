import scipy
from Quaternion import Quat
from euclid3 import Quaternion, Vector3

__author__ = 'edwardharry'


class quaternion(object):
    def __init__(self, v):

        self._Quat = Quat(v)

    class quaternionList(object):
        def __init__(self, quatList):
            self.quatList = quatList
            self.n = len(quatList)

        def __mul__(self, other):

            if type(other) is scipy.ndarray:
                n = other.shape[0]

                if n == 1:
                    return scipy.array([self.quatList[i] * other[0, :] for i in range(self.n)])

                elif self.n == 1:
                    return scipy.array([self.quatList[0] * other[i, :] for i in range(n)])

                elif self.n == n:
                    return scipy.array([self.quatList[i] * other[i, :] for i in range(n)])

                else:
                    print("quaternionList: sizes must match for multiple multiplications")
                    return None

            elif type(other) is quaternion:
                return quaternion.quaternionList([self.quatList[i] * other for i in range(self.n)])

            elif type(other) is quaternion.quaternionList:

                if self.n == 1:
                    return quaternion.quaternionList([self.quatList[0] * other.quatList[i] for i in range(other.n)])

                elif other.n == 1:
                    return quaternion.quaternionList([self.quatList[i] * other.quatList[0] for i in range(self.n)])

                elif self.n == other.n:
                    return quaternion.quaternionList([self.quatList[i] * other.quatList[i] for i in range(self.n)])

                else:
                    print("quaternionList: sizes must match for multiple multiplications")
                    return None

            else:
                return quaternion.quaternionList([self.quatList[i] * other for i in range(self.n)])

        def __truediv__(self, other):

            if type(other) is quaternion:
                return quaternion.quaternionList([self.quatList[i] / other for i in range(self.n)])

            elif type(other) is quaternion.quaternionList:

                if self.n == 1:
                    return quaternion.quaternionList([self.quatList[0] / other.quatList[i] for i in range(other.n)])

                elif other.n == 1:
                    return quaternion.quaternionList([self.quatList[i] / other.quatList[0] for i in range(self.n)])

                elif self.n == other.n:
                    return quaternion.quaternionList([self.quatList[i] / other.quatList[i] for i in range(self.n)])

                else:
                    print("quaternionList: sizes must match for multiple multiplications")
                    return None

            else:
                return quaternion.quaternionList([self.quatList[i] / other for i in range(self.n)])

        def __imul__(self, other):

            if type(other) is quaternion:

                for i in range(self.n):
                    self.quatList[i] *= other

            elif type(other) is quaternion.quaternionList:

                if self.n == 1:
                    self.quatList = [self.quatList[0] * other.quatList[i] for i in range(other.n)]
                    self.n = len(self.quatList)

                elif other.n == 1:
                    self.quatList = [self.quatList[i] * other.quatList[0] for i in range(self.n)]

                elif self.n == other.n:

                    for i in range(self.n):
                        self.quatList[i] *= other.quatList[i]

                else:
                    print("quaternionList: sizes must match for multiple multiplications")
                    return None

            else:

                for i in range(self.n):
                    self.quatList[i] *= other

        def __itruediv__(self, other):

            if type(other) is quaternion:

                for i in range(self.n):
                    self.quatList[i] /= other

            elif type(other) is quaternion.quaternionList:

                if self.n == 1:
                    self.quatList = [self.quatList[0] / other.quatList[i] for i in range(other.n)]
                    self.n = len(self.quatList)

                elif other.n == 1:
                    self.quatList = [self.quatList[i] / other.quatList[0] for i in range(self.n)]

                elif self.n == other.n:

                    for i in range(self.n):
                        self.quatList[i] /= other.quatList[i]

                else:
                    print("quaternionList: sizes must match for multiple multiplications")
                    return None

            else:

                for i in range(self.n):
                    self.quatList[i] /= other

    @classmethod
    def randRot(cls, n):
        from Utilities.maths.normalisation import normList

        tmp = normList(scipy.randn(n, 4), True)[1]
        tmp[tmp[:, 3] < 0, :] *= -1

        return quaternion.quaternionList([quaternion(tmp[i, :]) for i in range(n)])

    @classmethod
    def angAxis2e(cls, angle, axis):
        s = scipy.sin(angle / 2)
        vn = scipy.sqrt(scipy.sum(axis ** 2))

        if vn == 0:

            if s == 0:
                c = 0

            else:
                c = 1

            axis = scipy.zeros(3)

        else:
            c = scipy.cos(angle / 2)
            axis /= vn

        eout = scipy.concatenate((s * axis, [c]))

        if eout[3] < 0 and scipy.mod(angle / (2 * scipy.pi), 2) != 1:
            eout *= -1

        return eout

    @classmethod
    def angleAxis(cls, angle, axis):
        n1 = len(angle)
        n2 = axis.shape[0]

        if n1 != 1 and n2 != 1 and n1 != n2:
            print("angle and axis must have compatable sizes")
            return None

        n = scipy.maximum(n1, n2)
        d = scipy.zeros((n, 4))

        for i in range(n):
            d[i, :] = quaternion.angAxis2e(angle[scipy.minimum(i, n1 - 1)], axis[scipy.minimum(i, n2 - 1), :])

        return quaternion.quaternionList([quaternion(d[i, :]) for i in range(n)])

    def __mul__(self, other):

        if type(other) is scipy.ndarray:
            self.__convertQuatToQuaternion__()
            return self.__vectorMultiply__(other)

        elif type(other) is quaternion:
            return quaternion(self._Quat * other._Quat)

        else:
            return quaternion(self._Quat * other)

    def __truediv__(self, other):

        if type(other) is quaternion:
            return quaternion(self._Quat / other._Quat)

        else:
            return quaternion(self._Quat / other)

    def __imul__(self, other):

        if type(other) is quaternion:
            self._Quat *= other._Quat

        else:
            self._Quat *= other

    def __itruediv__(self, other):

        if type(other) is quaternion:
            self._Quat = self._Quat / other._Quat

        else:
            self._Quat = self._Quat / other


    def __vectorMultiply__(self, vec: scipy.ndarray):

        vec = vec.astype(scipy.float_)

        if len(vec.shape) == 1:
            vec = vec[scipy.newaxis, :]
            was1D = True

        else:
            was1D = False

        if vec.shape[0] == 3:
            isT = True
            vec = vec.transpose()
        else:
            isT = False

        for i in range(vec.shape[0]):
            vec[i, :] = self.__vectorMultiply_single__(vec[i, :])

        if isT:
            vec = vec.transpose()

        if was1D:
            vec = vec[0, :]

        return vec

    def __vectorMultiply_single__(self, vec: scipy.ndarray):

        self.__convertNPtoVector3__(vec)
        self.__convertVector3toNP__(self._Quaternion * self._vector3)
        return self._np

    def __convertQuatToQuaternion__(self):

        q = self._Quat.q
        self._Quaternion = Quaternion(q[3], q[0], q[1], q[2])

    def __convertNPtoVector3__(self, np):

        self._vector3 = Vector3(np[0], np[1], np[2])

    def __convertVector3toNP__(self, vector3):

        self._np = scipy.array([vector3[0], vector3[1], vector3[2]])

