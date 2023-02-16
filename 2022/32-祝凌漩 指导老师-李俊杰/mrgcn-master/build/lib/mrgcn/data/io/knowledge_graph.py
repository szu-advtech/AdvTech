#!/usr/bin/python3

from collections import Counter, defaultdict
from functools import total_ordering
import logging
import gzip
from tarfile import DEFAULT_FORMAT

from numpy import infty

from rdflib.graph import Graph
from rdflib.term import BNode, Literal
from rdflib.util import guess_format

from mrgcn.data.utils import is_gzip, is_readable


class KnowledgeGraph:
    """ Knowledge Graph Class
    A wrapper around an imported rdflib.Graph object with convenience functions
    """
    graph = None
    _property_distribution = {}

    def __init__(self, graph=None):
        self.logger = logging.getLogger()
        self.logger.debug("Initiating Knowledge Graph")

        if graph is not None:
            if type(graph) is Graph:
                self.graph = graph
            elif type(graph) is str:
                self.graph = self._read([graph])
            elif type(graph) is list:
                self.graph = self._read(graph)
            else:
                raise TypeError(":: Wrong input type: {}; requires path to RDF"
                                " graph or rdflib.graph.Graph object".format(type(graph)))
        else:
            self.graph = Graph()

        self._property_distribution = Counter(self.graph.predicates())
        self.logger.debug("Knowledge Graph ({} facts) succesfully imported".format(len(self.graph)))

    def _read(self, paths=None):
        graph = Graph()
        for path in paths:
            assert is_readable(path)
            if not is_gzip(path):
                graph.parse(path, format=guess_format(path))
            else:
                self.logger.debug("Input recognized as gzip file")
                with gzip.open(path, 'rb') as f:
                    graph.parse(f, format=guess_format(path[:-3]))

        return graph

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.graph.destroy("store")
        self.graph.close(True)

    def __len__(self):
        return len(self.graph)

    ### Generators ###

    def atoms(self, separate_literals=True):
        self.logger.debug("Yielding atoms (separated literals: {})".format(
            separate_literals))
        seen = set()
        for s, p, o in self.graph.triples((None, None, None)):
            for atom in (s, o):
                if separate_literals and isinstance(atom, Literal):
                    atom = self.UniqueLiteral(s, p, atom)
                if atom in seen:
                    continue
                seen.add(atom)

                yield atom

    def non_terminal_atoms(self):
        self.logger.debug("Yielding non-terminal atoms")
        for atom in frozenset(self.graph.subjects()):
            yield(atom)

    def terminal_atoms(self):
        self.logger.debug("Yielding terminal atoms")
        non_terminal_atoms = list(self.non_terminal_atoms())
        for atom in list(self.graph.objects()):
            if atom in non_terminal_atoms:
                continue

            yield(atom)

    def attributes(self):
        self.logger.debug("Yielding attributes")
        for obj in self.graph.objects():
            if type(obj) is Literal:
                yield(obj)

    def entities(self, omit_blank_nodes=False):
        self.logger.debug("Yielding entities")
        for res in self.atoms():
            if (type(res) is Literal or
               (omit_blank_nodes and type(res) is BNode)):
                continue

            yield(res)

    def objecttype_properties(self):
        # return unique properties
        attributes = frozenset(self.attributes())
        self.logger.debug("Yielding OT predicates")
        for p in self.graph.predicates():
            if len(set(self.graph.objects(None, p))-attributes) <= 0:
                # p is only used with a literal as object
                continue

            yield(p)

    def datatype_properties(self):
        # return unique properties
        objecttype_properties = set(self.objecttype_properties())
        self.logger.debug("Yielding DT predicates")
        for p in self.graph.predicates():
            if p in objecttype_properties:
                continue

            yield(p)

    def properties(self):
        self.logger.debug("Yielding properties")
        for p in self.graph.predicates():
            yield(p)

    def triples(self, triple=(None, None, None), separate_literals=True):
        self.logger.debug("Yielding triples (triple {})".format(triple))
        for s,p,o in self.graph.triples(triple):
            if separate_literals and isinstance(o, Literal):
                o = self.UniqueLiteral(s, p, o)
            yield s, p, o

    ## Statistics
    def property_frequency(self, property=None):
        if property is None:
            return self._property_distribution
        elif property in self._property_distribution:
            return self._property_distribution[property]

    def attribute_frequency(self, property, limit=None):
        attribute_freq = Counter(self.graph.objects(None, property))
        if limit is None:
            return attribute_freq.most_common()
        else:
            return attribute_freq.most_common(limit)

    ## Operators
    def sample(self, strategy=None, **kwargs):
        """ Sample this graph using the given strategy
        returns a KnowledgeGraph instance
        """
        if strategy is None:
            raise ValueError('Strategy cannot be left undefined')

        self.logger.debug("Sampling graph")
        return strategy.sample(self, **kwargs)

    def quickSort(self, lst):
        """Needed to sort deterministically when using UniqueLiterals"""
        less = list()
        pivotList = list()
        more = list()

        if len(lst) <= 1:
            return lst

        pivot = lst[0]
        for member in lst:
            if str(member) < str(pivot):
                less.append(member)
            elif str(member) > str(pivot):
                more.append(member)
            else:
                pivotList.append(member)

        less = self.quickSort(less)
        more = self.quickSort(more)

        return less + pivotList + more

    class UniqueLiteral(Literal):
        # literal with unique hash, irrespective of content
        def __new__(cls, s, p, o):
            self = super().__new__(cls, str(o), o.language, o.datatype, normalize=None)
            self.s = str(s)
            self.p = str(p)

            return self

        def __hash__(self):
            base = self.s + self.p + str(self)
            for attr in [self.language, self.datatype]:
                if attr is not None:
                    base += str(attr)

            return hash(base)

        def __eq__(self, other):
            if type(other) is not type(self):
                return False
            return hash(repr(self)) == hash(repr(other))

        @total_ordering
        def __lt__(self, other):
            if type(other) is not type(self):
                return False

            if str(self) < str(other):
                return True
            if self.s < other.s:
                return True
            if self.p < other.p:
                return True

            return False

if __name__ == "__main__":
    print("Knowledge Graph")
