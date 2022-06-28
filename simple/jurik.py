import numpy as np
from numba import njit


@njit(nogil=True)
def JRSX(SrcA: np.array, APeriod: int) -> np.array:
    f0 = f88 = f90 = v14 = v20 = f8 = f18 = f20 = f28 = f30 = f38 = f40 = f48 = f50 = f58 = f60 = f68 = f70 = f78 = f80 = 0
    Result = np.zeros_like(SrcA)
    for Bar in range(len(SrcA)):
        if f90 == 0:
            f90 = 1
            f0 = 0
            if APeriod - 1 >= 5:
                f88 = APeriod - 1
            else:
                f88 = 5

            f8 = 100 * SrcA[Bar]
            f18 = 3 / (APeriod + 2)
            f20 = 1 - f18
        else:
            if f88 <= f90:
                f90 = f88 + 1
            else:
                f90 = f90 + 1

            f10 = f8
            f8 = 100 * SrcA[Bar]
            v8 = f8 - f10
            f28 = f20 * f28 + f18 * v8
            f30 = f18 * f28 + f20 * f30
            vC = f28 * 1.5 - f30 * 0.5
            f38 = f20 * f38 + f18 * vC
            f40 = f18 * f38 + f20 * f40
            v10 = f38 * 1.5 - f40 * 0.5
            f48 = f20 * f48 + f18 * v10
            f50 = f18 * f48 + f20 * f50
            v14 = f48 * 1.5 - f50 * 0.5
            f58 = f20 * f58 + f18 * abs(v8)
            f60 = f18 * f58 + f20 * f60
            v18 = f58 * 1.5 - f60 * 0.5
            f68 = f20 * f68 + f18 * v18
            f70 = f18 * f68 + f20 * f70
            v1C = f68 * 1.5 - f70 * 0.5
            f78 = f20 * f78 + f18 * v1C
            f80 = f18 * f78 + f20 * f80
            v20 = f78 * 1.5 - f80 * 0.5
            if (f88 >= f90) and (f8 != f10):
                f0 = 1
            if (f88 == f90) and (f0 == 0):
                f90 = 0

        if (f88 < f90) and (v20 > 1.0e-10):
            v4 = (v14 / v20 + 1) * 50
            if v4 > 100:
                v4 = 100
            if v4 < 0:
                v4 = 0
        else:
            v4 = 50

        Result[Bar] = v4

    return Result


@njit(nogil=True)
def JCFBaux(SrcA: np.array, Depth: int) -> np.array:
    jrc04 = jrc05 = jrc06 = 0
    Result = np.zeros_like(SrcA)
    IntA = np.zeros_like(SrcA)
    for Bar in range(len(SrcA)):
        IntA[Bar] = abs(SrcA[Bar] - SrcA[Bar-1])

    for Bar in range(Depth, len(SrcA)-1):
        if Bar <= Depth * 2:
            jrc04 = jrc05 = jrc06 = 0

            for k in range(Depth):
                jrc04 = jrc04 + abs(SrcA[Bar-k] - SrcA[Bar-k-1])
                jrc05 = jrc05 + (Depth - k) * abs(SrcA[Bar-k] - SrcA[Bar-k-1])
                jrc06 = jrc06 + SrcA[Bar-k-1]

        else:
            jrc05 = jrc05 - jrc04 + IntA[Bar] * Depth
            jrc04 = jrc04 - IntA[Bar-Depth] + IntA[Bar]
            jrc06 = jrc06 - SrcA[Bar-Depth-1] + SrcA[Bar-1]

        jrc08 = abs(Depth * SrcA[Bar] - jrc06)
        Result[Bar] = 0 if jrc05 == 0 else jrc08 / jrc05

    return Result


@njit(nogil=True)
def JCFB(SrcA: np.array, degree: int, Smooth: int) -> np.array:
    """
    Dominant cycle analysis is a popular way to measure the strength of a trend,
    but it has obvious flaws. For example, what if no cycle exists in the data?
    We replaced cycle analysis (FFT and MESA) with a form of fractal analysis that
    works even when no cycles exist.
    The tool is called CFB, Composite Fractal Behavior Index.
    """

    Result = np.zeros_like(SrcA)
    lags = np.cumsum(np.array(sorted([np.power(2, p) for p in range(degree)] * 2)))
    lags = np.hstack((np.array([2]), lags + 2))[:-1]
    cnt = len(lags)
    X = np.zeros(cnt)
    E = np.zeros(cnt)
    result = 20
    Result[0] = result
    AUX = [JCFBaux(SrcA, lag) for lag in lags]

    for Bar in range(1, len(SrcA)):
        if Bar <= Smooth:
            E = np.zeros(cnt)
            for j in range(Bar):
                for k in range(cnt):
                    E[k] += AUX[k][Bar - j]
            for k in range(cnt):
                E[k] /= Bar
        else:
            for k in range(cnt):
                E[k] += (AUX[k][Bar] - AUX[k][Bar - Smooth]) / Smooth

        if Bar > 5:
            a = 1
            for k in np.arange(cnt-1, 0, -2) - 1:  # odd indexes from high to low
                X[k] = a * E[k]
                a *= 1 - X[k]

            b = 1
            for k in np.arange(cnt-1, 0, -2):   # even indexes from high to low
                X[k] = b * E[k]
                b *= 1 - X[k]

            sqw = sum([X[k] * X[k] * lags[k] for k in range(len(lags))])
            sq = sum(np.square(X))
            result = 0 if sq == 0 else sqw / sq

        Result[Bar] = result

    return Result


@njit(nogil=True)
def JVELaux1(SrcA: np.array, Depth: int) -> np.array:
    Result = np.zeros_like(SrcA)

    w = Depth + 1
    jv1 = w * (w + 1) / 2
    jv2 = jv1 * (2 * w + 1) / 3
    jv3 = jv1 * jv1 * jv1 - jv2 * jv2

    for Bar in range(Depth, len(SrcA)):
        s = sq = 0
        for k in range(w):
            s += SrcA[Bar - k] * (w - k)
            sq += SrcA[Bar - k] * (w - k) * (w - k)

        Result[Bar] = (sq * jv1 - s * jv2) / jv3

    return Result


@njit(nogil=True)
def JTPO(SrcA: np.array, Period: int) -> np.array:
    maxPeriod = 8000
    period = maxPeriod if Period > maxPeriod else Period

    Result = np.zeros_like(SrcA)
    f10 = f18 = f20 = f30 = f38 = f40 = f48 = 0

    arr0 = np.zeros(maxPeriod)
    arr1 = np.zeros(maxPeriod)
    arr2 = np.zeros(maxPeriod)
    arr3 = np.zeros(maxPeriod)

    for Bar in range(len(SrcA)):
        if f38 == 0:
            f38 = 1
            f40 = 0
            f30 = period - 1 if period - 1 >= 2 else 2
            f48 = f30 + 1
            f10 = SrcA[Bar]
            arr0[f38] = SrcA[Bar]
            f18 = 12 / (f48 * (f48 - 1) * (f48 + 1))
            f20 = (f48 + 1) * 0.5

        else:
            f38 = f38 + 1 if f38 <= f48 else f48 + 1
            f8 = f10
            f10 = SrcA[Bar]

            if f38 > f48:
                for k in range(2, f48+1):
                    arr0[k - 1] = arr0[k]
                arr0[f48] = SrcA[Bar]
            else:
                arr0[f38] = SrcA[Bar]

            if (f30 >= f38) and (f8 != f10):
                f40 = 1
            if (f30 == f38) and (f40 == 0):
                f38 = 0

        if f38 >= f48:
            for j in range(1, f48+1):
                arr2[j] = j
                arr3[j] = j
                arr1[j] = arr0[j]

            for j in range(1, f48):
                var24 = arr1[j]
                var12 = j
                for k in range(j + 1, f48 + 1):
                    if arr1[k] < var24:
                        var24 = arr1[k]
                        var12 = k

                var20 = arr1[j]
                arr1[j] = arr1[var12]
                arr1[var12] = var20
                var20 = arr2[j]
                arr2[j] = arr2[var12]
                arr2[var12] = var20

            j = 1
            while f48 > j:
                k = j + 1
                var14 = 1
                var1C = arr3[j]

                while var14 != 0:
                    if arr1[j] != arr1[k]:
                        if k - j > 1:
                            var1C /= k - j
                            for n in range(j, k):
                                arr3[n] = var1C
                        var14 = 0
                    else:
                        var1C += arr3[k]
                        k += 1
                j = k

            var1C = 0
            for j in range(1, f48 + 1):
                var1C += (arr3[j] - f20) * (arr2[j] - f20)

            result = f18 * var1C
        else:
            result = 0

        Result[Bar] = result

    return Result


@njit(nogil=True)
def JMA(src, period: int = 7, phase: float = 50, power: int = 2):
    Result = np.copy(src)

    phaseRatio = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100 + 1.5)
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = pow(beta, power)

    e0 = np.zeros_like(src)
    e1 = np.zeros_like(src)
    e2 = np.zeros_like(src)

    for i in range(1, src.shape[0]):
        e0[i] = (1 - alpha) * src[i] + alpha * e0[i-1]
        e1[i] = (src[i] - e0[i]) * (1 - beta) + beta * e1[i-1]
        e2[i] = (e0[i] + phaseRatio * e1[i] - Result[i - 1]) * pow(1 - alpha, 2) + pow(alpha, 2) * e2[i - 1]
        Result[i] = e2[i] + Result[i - 1]

    Result[:period*2] = Result[period*2]
    return Result
