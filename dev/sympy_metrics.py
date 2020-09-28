import numpy as np
import sympy

sqrt = sympy.sqrt

tp, tn, fp, fn, B = sympy.symbols(['tp', 'tn', 'fp', 'fn', 'B'], integer=True, negative=False)
numer = (tp * tn - fp * fn)
denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
mcc = numer / denom

ppv = tp / (tp + fp)
tpr = tp / (tp + fn)

FM = ((tp / (tp + fn)) * (tp / (tp + fp))) ** 0.5

B2 = (B ** 2)
B2_1 = (1 + B2)

F_beta_v1 = (B2_1 * (ppv * tpr)) / ((B2 * ppv) + tpr)
F_beta_v2 = (B2_1 * tp) / (B2_1 * tp + B2 * fn + fp)

# Demo how Beta interacts with harmonic mean weights for F-Beta
w1 = 1
w2 = 1
x1 = tpr
x2 = ppv
harmonic_mean = (w1 + w2) / ((w1 / x1) + (w2 / x2))
harmonic_mean = sympy.simplify(harmonic_mean)
expr = sympy.simplify(harmonic_mean - F_beta_v1)
sympy.solve(expr, B2)

geometric_mean = ((x1 ** w1) * (x2 ** w2)) ** (1 / (w1 + w2))
geometric_mean = sympy.simplify(geometric_mean)
assert sympy.simplify(sympy.simplify(geometric_mean) - sympy.simplify(FM)) == 0

print('geometric_mean = {!r}'.format(geometric_mean))

# How do we apply weights to precision and recall when tn is included
# in mcc?

print(sympy.simplify(F_beta_v1))
print(sympy.simplify(F_beta_v2))
assert sympy.simplify(sympy.simplify(F_beta_v1) - sympy.simplify(F_beta_v2)) == 0


tnr_denom = (tn + fp)
tnr = tn / tnr_denom

pnv_denom = (tn + fn)
npv = tn / pnv_denom

mk = ppv + npv - 1  # markedness (precision analog)
bm = tpr + tnr - 1  # informedness (recall analog)

# Demo how Beta interacts with harmonic mean weights for F-Beta
w1 = 2
w2 = 1
x1 = mk  # precision analog
x2 = bm  # recall analog
geometric_mean = ((x1 ** w1) * (x2 ** w2)) ** (1 / (w1 + w2))
geometric_mean = sympy.simplify(geometric_mean)
print('geometric_mean w1=2 = {!r}'.format(geometric_mean))

w1 = 0.5
w2 = 1
geometric_mean = ((x1 ** w1) * (x2 ** w2)) ** (1 / (w1 + w2))
geometric_mean = sympy.simplify(geometric_mean)
print('geometric_mean w1=.5 = {!r}'.format(geometric_mean))

# By taking the weighted geometric mean of bm and mk we can effectively
# create a mcc-beta measure

values = {fn: 3, fp: 10, tp: 100, tn: 200}

# Cant seem to verify that gmean(bm, mk) == mcc, with sympy, but it is true
values = {fn: np.random.rand() * 10, fp: np.random.rand() * 10, tp: np.random.rand() * 10, tn: np.random.rand() * 10}
print(geometric_mean.subs(values))
print(abs(mcc).subs(values))

delta = sympy.simplify(sympy.simplify(geometric_mean) - sympy.simplify(sympy.functions.Abs(mcc)))
# assert sympy.simplify(sympy.simplify(geometric_mean) - sympy.simplify(sympy.functions.Abs(mcc))) == 0
