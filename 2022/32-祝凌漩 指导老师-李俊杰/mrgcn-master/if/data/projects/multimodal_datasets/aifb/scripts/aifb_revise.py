#!/usr/bin/env python

from sys import stdin

from rdflib import Literal, Graph, URIRef
from rdflib.namespace import XSD


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def main(g):
    for s, p, o in g.triples((None, None, None)):
        if p == URIRef('http://swrc.ontoware.org/ontology#year'):
            g.remove((s, p, o))
            g.add((s, p, Literal(o.value[:4], datatype=XSD.gYear)))

        if p == URIRef('http://swrc.ontoware.org/ontology#number'):
            if not is_number(o.value):
                continue
            g.remove((s, p, o))
            g.add((s, p, Literal(o.value, datatype=XSD.decimal)))

if __name__ == "__main__":
    g = Graph()
    g.parse(data=stdin.read(), format='nt')

    main(g)
    g.serialize('./out.nt', format='nt')
