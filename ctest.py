import ctypes
import itertools
import numpy as np

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

bez_fit_lib = ctypes.CDLL('./bezier_fit.so')

point_ptr = (ctypes.c_double * 1024)()
bezier_ptr = (ctypes.c_double * 1024)()
mlFeature_ptr = (ctypes.c_double * (1024 * 3) )()
c_fitCurve = wrap_function(bez_fit_lib,"c_FitCurve",ctypes.c_int,[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]);
c_mlEncode = wrap_function(bez_fit_lib,"c_ML_EncodeCurves",None,[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double)]);
def fitCurve(coords,error=6.0):
    point_ptr[:len(coords)*2] = list(itertools.chain(*coords))
    n_beziers = c_fitCurve(point_ptr,len(coords),error*error,bezier_ptr)
    c_mlEncode(bezier_ptr, n_beziers, mlFeature_ptr)
    return np.array(mlFeature_ptr[:9*n_beziers]).reshape((-1,9))
    
    # np.array()
np.set_printoptions(suppress=True)

points = [
    [ 591, 133 ],
    [ 586, 129 ],
    [ 577, 125 ],
    [ 564, 122 ],
    [ 554, 121 ],
    [ 546, 120 ],
    [ 532, 120 ],
    [ 516, 120 ],
    [ 503, 122 ],
    [ 493, 123 ],
    [ 481, 129 ],
    [ 474, 131 ],
    [ 466, 137 ],
    [ 461, 142 ],
    [ 453, 152 ],
    [ 451, 158 ],
    [ 446, 173 ],
    [ 443, 189 ],
    [ 442, 206 ],
    [ 442, 221 ],
    [ 443, 232 ],
    [ 451, 246 ],
    [ 482, 279 ],
    [ 523, 305 ],
    [ 572, 333 ],
    [ 614, 350 ],
    [ 631, 358 ],
    [ 641, 366 ],
    [ 645, 380 ],
    [ 648, 394 ],
    [ 648, 421 ],
    [ 617, 460 ],
    [ 566, 488 ],
    [ 475, 513 ],
    [ 380, 520 ],
    [ 294, 506 ],
    [ 256, 487 ],
    ];

print(fitCurve(points))