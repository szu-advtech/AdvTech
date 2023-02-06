#!/usr/bin/env python

from queue import Queue


class XSDHierarchy:
    _root = None
    _index = None

    def __init__(self):
        self._build_hierarchy()
        self._build_index()

    def _build_index(self):
        self._index = dict()
        if self._root is None:
            return

        self._index[self._root.name] = self._root
        q = Queue()
        for child in self._root.children:
            q.put(child)

        while not q.empty():
            node = q.get()
            self._index[node.name] = node

            for child in node.children:
                q.put(child)

    def _build_hierarchy(self):
        self._root = self.XSDDatatype("anyType",
                                      None)

        anySimpleType = self.XSDDatatype("anySimpleType",
                                         self._root)
        self._root.children = {
            anySimpleType,
            self.XSDDatatype("ENTITIES",
                             self._root),
            self.XSDDatatype("IDREFS",
                             self._root),
            self.XSDDatatype("NMTOKENS",
                             self._root)}

        anyAtomicType = self.XSDDatatype("anyAtomicType",
                                         anySimpleType)
        anySimpleType.children = { anyAtomicType }

        dateTime = self.XSDDatatype("dateTime",
                                     anyAtomicType)
        decimal = self.XSDDatatype("decimal",
                                    anyAtomicType)
        double = self.XSDDatatype("double",
                                   anyAtomicType)
        duration = self.XSDDatatype("duration",
                                     anyAtomicType)
        floattype = self.XSDDatatype("float",
                                      anyAtomicType)
        string = self.XSDDatatype("string",
                                   anyAtomicType)
        anyAtomicType.children = {
            self.XSDDatatype("anyURI",
                             anyAtomicType),
            self.XSDDatatype("base64Binary",
                             anyAtomicType),
            self.XSDDatatype("boolean",
                             anyAtomicType),
            self.XSDDatatype("date",
                             anyAtomicType),
            dateTime,
            decimal,
            double,
            duration,
            floattype,
            self.XSDDatatype("duration",
                             anyAtomicType),
            self.XSDDatatype("gDay",
                             anyAtomicType),
            self.XSDDatatype("gMonth",
                             anyAtomicType),
            self.XSDDatatype("gMonthDay",
                             anyAtomicType),
            self.XSDDatatype("gYear",
                             anyAtomicType),
            self.XSDDatatype("gYearMonth",
                             anyAtomicType),
            self.XSDDatatype("hexBinary",
                             anyAtomicType),
            self.XSDDatatype("QName",
                             anyAtomicType),
            string,
            self.XSDDatatype("time",
                             anyAtomicType)}

        dateTime.children = { self.XSDDatatype("dateTimeStamp",
                                               dateTime) }

        # integers
        integer = self.XSDDatatype("integer", decimal)
        longtype = self.XSDDatatype("long", integer)
        inttype = self.XSDDatatype("int", longtype)
        short = self.XSDDatatype("short", inttype)
        byte = self.XSDDatatype("byte", short)
        nonNegativeInteger = self.XSDDatatype("nonNegativeInteger", integer)
        positiveInteger = self.XSDDatatype("positiveInteger", nonNegativeInteger)
        unsignedLong = self.XSDDatatype("unsignedLong", nonNegativeInteger)
        unsignedInt = self.XSDDatatype("unsignedInt", unsignedLong)
        unsignedShort = self.XSDDatatype("unsignedShort", unsignedInt)
        unsignedByte = self.XSDDatatype("unsignedByte", unsignedShort)
        nonPositiveInteger = self.XSDDatatype("nonPositiveInteger", integer)
        negativeInteger = self.XSDDatatype("negativeInteger", nonPositiveInteger)

        short.children = { byte }
        inttype.children = { short }
        longtype.children = { inttype }
        unsignedShort.children = { unsignedByte }
        unsignedInt.children = { unsignedShort }
        unsignedLong.children = { unsignedInt }
        nonNegativeInteger.children = { unsignedLong,
                                        positiveInteger }
        nonPositiveInteger.children = { negativeInteger }
        integer.children = { longtype,
                             nonNegativeInteger,
                             nonPositiveInteger }
        decimal.children = { integer }

        dayTimeDuration = self.XSDDatatype("dayTimeDuration",
                                           duration)
        yearMonthDuration = self.XSDDatatype("yearMonthDuration",
                                             duration)
        duration.children = { dayTimeDuration,
                              yearMonthDuration }

        # strings
        normalizedString = self.XSDDatatype("normalizedString",
                                             string)
        token = self.XSDDatatype("token",
                                 normalizedString)
        language = self.XSDDatatype("language",
                                    token)
        Name = self.XSDDatatype("Name",
                                token)
        NCName = self.XSDDatatype("NCName",
                                  Name)
        ENTITY = self.XSDDatatype("ENTITY",
                                  NCName)
        ID = self.XSDDatatype("ID",
                              NCName)
        IDREF = self.XSDDatatype("IDREF",
                                 NCName)
        NMTOKEN = self.XSDDatatype("NMTOKEN",
                                   token)

        NCName.children = { ENTITY,
                            ID,
                            IDREF }
        Name.children = { NCName }
        token.children = { language,
                           Name,
                           NMTOKEN }
        normalizedString.children = { token }
        string.children = { normalizedString }

        # special type
        numeric = self.XSDDatatype("numeric",
                                   anyAtomicType,
                                   { decimal,
                                     double,
                                     floattype })
        anyAtomicType.children.add(numeric)

        decimal.parent = numeric
        double.parent = numeric
        floattype.parent = numeric

    def parentof(self, a, b):
        # true if b is parent of a
        return self._index[a].parent is self._index[b]

    def subtypeof(self, a, b):
        # true iff a is b or b in tree rooted at a
        a = self._index[a]
        b = self._index[b]
        if a is b or a is self._root:
            return True

        while b is not self._root:
            b = b.parent
            if a is b:
                return True

        return False

    class XSDDatatype:
        name = ""
        parent = None
        children = None

        def __init__(self, name, parent, children=set()):
            self.name = name
            self.parent = parent
            self.children = children
