# Entropy in (16 ** 8)
"""
Entropy check, how big of a file name do we realistically need?

# !pip install numexpr

import numexpr
keys = [
    # Possible hex variants
    '16. **  8',
    '16. ** 16',
    '16. ** 32',
    # Possible abc variants
    '26. **  8',
    '26. ** 14',
    '26. ** 16',
    '26. ** 28',
    '26. ** 32',
]
candidates = {}
for k in keys:
    candidates[k] = numexpr.evaluate(k)

candidates = {k: numexpr.evaluate(k) for k in candidates}
entropy = {k: numexpr.evaluate('log({})'.format(k)) for k in candidates}
_bits = {k: numexpr.evaluate('log({}) / log(2)'.format(k)) for k in candidates}

_bits = ub.sorted_vals(_bits)
entropy = ub.sorted_vals(entropy)
candidates = ub.sorted_vals(candidates)

bits = ub.odict()
for k, v in _bits.items():
    if k.startswith('16'):
        k = ub.color_text(k, 'blue')
    else:
        k = ub.color_text(k, 'green')
    bits[k] = v

print('candidates = {}'.format(ub.repr2(candidates, nl=1, precision=4)))
print('entropy = {}'.format(ub.repr2(entropy, nl=1, precision=4)))
print('bits    = {}'.format(ub.repr2(bits, nl=1, sk=1, precision=4)))

# We save the following number of chars by using abc format.


print(entropy['16. **  8'])
print(entropy['16. ** 16'])
print(entropy['16. ** 32'])

print(entropy['26. **  8'])
print(entropy['26. ** 14'])
print(entropy['26. ** 28'])
'''

     hex           |                 abc |  savings |
------------------ | ------------------- | -------- |
16 **  8 ~ exp(22) |  26 **  8 ~ exp(26) |        0 |
16 ** 16 ~ exp(44) |  26 ** 14 ~ exp(45) |        2 |
16 ** 32 ~ exp(88) |  26 ** 28 ~ exp(91) |        4 |

'''
If we want the accuracy of 16-bit hex hashes, we can save 2 characters by
# using the abc base .
"""
