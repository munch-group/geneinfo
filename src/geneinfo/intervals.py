# In all of the following, the list of intervals must be sorted and 
# non-overlapping. We also assume that the intervals are half-open, so
# that x is in tp(start, end) iff start <= x and x < end.

from functools import wraps, reduce, partial


def flatten(list_of_tps):
    """
    Convert a list of sorted intervals to a list of endpoints.
    :param query: Sorted list of (start, end) tuples.
    :type query: list
    :returns: A list of interval ends
    :rtype: list
    """
    return reduce(lambda ls, ival: ls + list(ival), list_of_tps, [])


def unflatten(list_of_endpoints):
    """
    Convert a list of sorted endpoints into a list of intervals.
    :param query: Sorted list of ends.
    :type query: list
    :returns: A list of intervals.
    :rtype: list
    """
    return [ [list_of_endpoints[i], list_of_endpoints[i + 1]]
          for i in range(0, len(list_of_endpoints) - 1, 2)]


def merge(query, annot, op):
    """
    Merge two lists of sorted intervals according to the boolean function op.
    :param query: List of (start, end) tuples.
    :type query: list
    :param query: List of (start, end) tuples.
    :type query: list
    :param op: Boolean function taking two a bolean arguments.
    :type op: function
    :returns: A list of interval merged according to op
    :rtype: list
    """
    a_endpoints = flatten(query)
    b_endpoints = flatten(annot)

    assert a_endpoints == sorted(a_endpoints), "not sorted or non-overlaping"
    assert b_endpoints == sorted(b_endpoints), "not sorted or non-overlaping"


    sentinel = max(a_endpoints[-1], b_endpoints[-1]) + 1
    a_endpoints += [sentinel]
    b_endpoints += [sentinel]

    a_index = 0
    b_index = 0

    res = []

    scan = min(a_endpoints[0], b_endpoints[0])
    while scan < sentinel:
        in_a = not ((scan < a_endpoints[a_index]) ^ (a_index % 2))
        in_b = not ((scan < b_endpoints[b_index]) ^ (b_index % 2))
        in_res = op(in_a, in_b)

        if in_res ^ (len(res) % 2):
            res += [scan]
        if scan == a_endpoints[a_index]: 
            a_index += 1
        if scan == b_endpoints[b_index]: 
            b_index += 1
        scan = min(a_endpoints[a_index], b_endpoints[b_index])

    return unflatten(res)

def diff(a, b):
    if not (a and b):
        return a and a or b
    return merge(a, b, lambda in_a, in_b: in_a and not in_b)

def union(a, b):
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a or in_b)

def intersect(a, b):
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a and in_b)

def collapse(a):
    a_union = [list(a[0])]
    for i in range(1, len(a)):
        x = a[i]
        if a_union[-1][1] < x[0]:
            a_union.append(list(x))
        else:
            a_union[-1][1] = x[1]
    return a_union

def invert(a, left, right):
    starts, ends = zip(*collapse(sorted(a)))
    
    assert left <= starts[0] and right >= ends[-1]    

    starts = list(starts)
    ends = list(ends)
        
    ends.insert(0, left)
    starts.append(right)

    # remove first and last interval if they are empty
    if starts[0] == ends[0]:
        del starts[0]
        del ends[0]
    if starts[-1] == ends[-1]: 
        del starts[-1]
        del ends[-1]            
    inverted = zip(ends, starts)
    inverted = list(map(tuple, inverted))
    return inverted
