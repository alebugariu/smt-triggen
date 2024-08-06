###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from functools import reduce

from sortedcontainers import SortedSet

from src.minimizer.exceptions import InvalidSmtFileException


def toposort(data):
    """
    Inspired by: https://rosettacode.org/wiki/Topological_sort#Python
    """
    assert len(data) > 0
    for k, v in data.items():
        v.discard(k)  # Ignore self dependencies
    extra_items_in_deps = reduce(set.union, data.values()) - SortedSet(data.keys())
    data.update({item: SortedSet() for item in extra_items_in_deps})
    while True:
        ordered = SortedSet(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        yield sorted(ordered)
        data = {item: (dep - ordered) for item, dep in data.items()
                if item not in ordered}
    if bool(data):
        raise InvalidSmtFileException("A cyclic dependency exists amongst %r" % data)


def get_connected_component(data, root):
    comp = SortedSet()

    def walk_data_rec(data, node):
        nonlocal comp
        comp.add(node)
        unvisited_children = {item for item, dep in data.items() if node in dep and item not in comp}
        for child in unvisited_children:
            walk_data_rec(data, child)

    walk_data_rec(data, root)
    return comp


def topo_key(topo_layers, item):
    try:
        return next(i for i, layer in enumerate(topo_layers) if item in layer)
    except StopIteration:
        return -1


def test_toposort(data):
    res = list(toposort(data))
    print('Topological sort resulted in the following layers: \n' + str(res))
    for root_nr, root in enumerate(res[0]):
        print(' Component #' + str(root_nr) + ':' + str(sorted(get_connected_component(data, root),
                                                        key=lambda x: topo_key(res, x))))


def main():

    testdata1 = {
        'des_system_lib': SortedSet('std synopsys std_cell_lib des_system_lib dw02 dw01 ramlib ieee'.split()),
        'dw01': SortedSet('ieee dw01 dware gtech'.split()),
        'dw02': SortedSet('ieee dw02 dware'.split()),
        'dw03': SortedSet('std synopsys dware dw03 dw02 dw01 ieee gtech'.split()),
        'dw04': SortedSet('dw04 ieee dw01 dware gtech'.split()),
        'dw05': SortedSet('dw05 ieee dware'.split()),
        'dw06': SortedSet('dw06 ieee dware'.split()),
        'dw07': SortedSet('ieee dware'.split()),
        'dware': SortedSet('ieee dware'.split()),
        'gtech': SortedSet('ieee gtech'.split()),
        'ramlib': SortedSet('std ieee'.split()),
        'std_cell_lib': SortedSet('ieee std_cell_lib'.split()),
        'synopsys': SortedSet(),
    }

    testdata2 = {
        'b': SortedSet('z'),
        'c': SortedSet('z'),
        'd': SortedSet('c z'.split()),
        'g': SortedSet('e'),
        'f': SortedSet('e'),
        'q': SortedSet('e')
    }

    test_toposort(testdata1)


if __name__ == '__main__':
    main()
