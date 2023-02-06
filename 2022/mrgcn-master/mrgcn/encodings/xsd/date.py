#!/usr/bin/python3

import logging
from math import pi, sin, cos
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{1,4})"  # only consider years from -9999 to 9999
_REGEX_MONTH_FRAG = "(?P<month>\d{1,2})"
_REGEX_DAY_FRAG = "(?P<day>\d{1,2})"
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"
_REGEX_DATE = "{}-{}-{}(?:{})?".format(_REGEX_YEAR_FRAG,
                                     _REGEX_MONTH_FRAG,
                                     _REGEX_DAY_FRAG,
                                     _REGEX_TIMEZONE_FRAG)
_DAY_RAD = 2*pi/31
_MONTH_RAD = 2*pi/12
_YEAR_DECADE_RAD = 2*pi/10

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD date literals

    Definition
    - date := yearFrag '-' monthFrag '-' dayFrag timezoneFrag?

    Encoding
    - a vector v of length C = 10
    -- v[0] : '-'? : BCE or AD; -1.0 if '-', else 1.0
                     Note: a. needed to represent difference 0YY AD and 0YY BCE
                           b. mapping assumes majority is AD
    -- v[1] : \d\d : centuries; numerical and normalized
                     Note: no separation between hundred and thousands as the
                           latter's range is typically limited
    -- v[2:4] : \d,\d   : decades on circle
    -- v[4:6] : \d,\d   : individual years on circle
    -- v[6:8]: \d,\d : point on circle representing month
    -- v[8:10]: \d,\d : point on circle representing day

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [enc, node_idx, None];
                    enc :- numpy array M x C;
                        M :- number of nodes with this feature, such that M <= N
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    None :- not used here

    """
    logger.debug("Generating date encodings")
    C = 10  # number of items per feature

    return generate_relationwise_features(nodes_map, node_predicate_map, C,
                                          config)

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    m = dict()
    encodings = dict()
    node_idx = dict()
    values_min = dict()
    values_max = dict()

    features = list(getFeature(nodes_map, XSD.date))
    n = len(features)

    failed = 0
    for node, i in features:
        try:
            value = validate(str(node))

            sign = 1. if value.group('sign') == '' else -1.
            year = value.group('year')

            # separate centuries, decades, and individual years
            separated = separate(year)

            c = int(separated.group('century'))

            decade = int(separated.group('decade'))
            dec1, dec2 = point(decade, _YEAR_DECADE_RAD)

            year = int(separated.group('year'))
            y1, y2 = point(year, _YEAR_DECADE_RAD)

            month = value.group('month')
            m1, m2 = point(int(month), _MONTH_RAD)

            day = value.group('day')
            d1, d2 = point(int(day), _DAY_RAD)
        except:
            failed += 1
            continue

        for p in node_predicate_map[node]:
            if p not in encodings.keys():
                encodings[p] = np.empty(shape=(n, C), dtype=np.float32)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0
                values_min[p] = None
                values_max[p] = None

            if values_max[p] is None or c > values_max[p]:
                values_max[p] = c
            if values_min[p] is None or c < values_min[p]:
                values_min[p] = c

            idx = m[p]
            # add to matrix structures
            encodings[p][idx] = [sign, c, dec1, dec2, y1, y2, m1, m2, d1, d2]
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique date encodings ({} failed)".format(msum,
                                                                         failed))

    if msum <= 0:
        return None

    # normalization over centuries
    for p in encodings.keys():
        idx = np.arange(m[p])
        if values_max[p] == values_min[p]:
            encodings[p][idx,1] = 0.0
            continue

        encodings[p][idx,1] = (2*(encodings[p][idx,1] - values_min[p]) /
                              (values_max[p] - values_min[p])) -1.0

    return [[encodings[p][:m[p]], node_idx[p][:m[p]], C*np.ones(m[p])]
            for p in encodings.keys()]

def point(m, rad):
    # place on circle
    return (sin(m*rad), cos(m*rad))

def separate(year):
    regex = "^(?P<century>\d{0,2}?)(?P<decade>\d?)(?P<year>\d)$"
    return match(regex, year)

def validate(value):
    return match(_REGEX_DATE, value)

def getFeature(nodes_map, datatype):
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(datatype):
            continue

        yield (node, i)
