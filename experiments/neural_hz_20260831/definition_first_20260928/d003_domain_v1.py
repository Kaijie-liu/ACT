"""Default-off, exact D003 shared-circuit reference; no solver or ACT admission.

All graph history is retained. A feasible evaluation is a mathematical witness,
not a validated network adversarial example. Limits are rejection boundaries.
"""

from dataclasses import dataclass
from fractions import Fraction


MAX_INPUTS = 16
MAX_FREE_BINARY = 8
MAX_BINARY = 64
MAX_NODES = 256
MAX_VECTOR_HANDLES = 256
MAX_PREDICATES = 128
MAX_FAN_IN = 32
MAX_AFFINE_EDGES = 4096
MAX_RATIONAL_BITS = 512


def _q(value):
    if type(value) not in (int, Fraction):
        raise TypeError("exact int or Fraction required (not bool or float)")
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise ValueError("rational exceeds 512-bit reference limit")
    return value


def _add(a, b):
    return _q(a + b)


def _mul(a, b):
    return _q(a * b)


def _neg(a):
    return _q(-a)


def _values(values, length):
    collected = []
    for index, value in enumerate(values):
        if index >= length:
            raise ValueError("wrong vector arity")
        collected.append(_q(value))
    if len(collected) != length:
        raise ValueError("wrong vector arity")
    return tuple(collected)


def _count(value, maximum):
    if type(value) is not int:
        raise TypeError("dimension must be an int")
    if not 0 <= value <= maximum:
        raise ValueError("dimension exceeds reference limit")
    return value


@dataclass(frozen=True, slots=True)
class Vector:
    owner: object
    nodes: tuple

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        selected = self.nodes[key]
        return Vector(self.owner, selected if isinstance(key, slice) else (selected,))


@dataclass(frozen=True, slots=True)
class Node:
    kind: str
    terms: tuple = ()
    constant: Fraction = Fraction(0)
    index: int = -1


@dataclass(frozen=True, slots=True)
class Gate:
    node: int
    preactivation: int
    bit: int


@dataclass(frozen=True, slots=True)
class Predicate:
    node: int
    relation: str
    rhs: Fraction


@dataclass(frozen=True, slots=True)
class Evaluation:
    feasible: bool
    node_values: tuple
    outputs: tuple
    inputs: tuple
    _owner: object
    _bits: tuple


class Builder:
    """One input frame; operations validate fully before an atomic append."""

    def __init__(self, n_inputs, n_binary):
        self.n_inputs = _count(n_inputs, MAX_INPUTS)
        self.n_binary = _count(n_binary, MAX_FREE_BINARY)
        self._owner = object()
        self._nodes = [Node("input", index=i) for i in range(n_inputs)]
        self._nodes += [Node("binary", index=i) for i in range(n_binary)]
        self._gates = []
        self._predicates = []
        self._edges = 0
        self._closed = False

    def _open(self):
        if self._closed:
            raise ValueError("builder is frozen")

    def _vector(self, vector):
        self._open()
        if not isinstance(vector, Vector) or vector.owner is not self._owner:
            raise ValueError("vector belongs to another input frame")
        if type(vector.nodes) is not tuple or len(vector.nodes) > MAX_VECTOR_HANDLES:
            raise ValueError("vector handle limit exceeded or invalid handle tuple")
        if any(
            type(i) is not int or not 0 <= i < len(self._nodes) for i in vector.nodes
        ):
            raise ValueError("invalid node handle")
        return vector.nodes

    def inputs(self):
        self._open()
        return Vector(self._owner, tuple(range(self.n_inputs)))

    def binary_inputs(self):
        self._open()
        return Vector(self._owner, tuple(range(self.n_inputs, self.n_inputs + self.n_binary)))

    def _capacity(self, nodes, edges=0, gates=0):
        if len(self._nodes) + nodes > MAX_NODES:
            raise ValueError("node limit exceeded")
        if self._edges + edges > MAX_AFFINE_EDGES:
            raise ValueError("affine edge limit exceeded")
        if self.n_binary + len(self._gates) + gates > MAX_BINARY:
            raise ValueError("total binary limit exceeded")

    def _append(self, nodes, edges=0, gates=()):
        self._capacity(len(nodes), edges, len(gates))
        first = len(self._nodes)
        self._nodes.extend(nodes)
        self._gates.extend(gates)
        self._edges += edges
        return Vector(self._owner, tuple(range(first, first + len(nodes))))

    def affine(self, vector, weights, bias):
        sources = self._vector(vector)
        if len(sources) > MAX_FAN_IN:
            raise ValueError("affine fan-in limit exceeded")
        rows, edges = [], 0
        for row in weights:
            self._capacity(len(rows) + 1, edges)
            row = _values(row, len(sources))
            edges += sum(bool(coefficient) for coefficient in row)
            self._capacity(len(rows) + 1, edges)
            rows.append(row)
        bias = _values(bias, len(rows))
        nodes = tuple(Node("affine", tuple((i, c) for i, c in zip(sources, row) if c), offset)
                      for row, offset in zip(rows, bias))
        return self._append(nodes, edges)

    def relu(self, vector):
        sources = self._vector(vector)
        self._capacity(len(sources), gates=len(sources))
        first, first_bit = len(self._nodes), self.n_binary + len(self._gates)
        nodes = tuple(Node("relu", ((source, Fraction(1)),), index=first_bit + i)
                      for i, source in enumerate(sources))
        gates = tuple(Gate(first + i, source, first_bit + i)
                      for i, source in enumerate(sources))
        return self._append(nodes, gates=gates)

    def add(self, left, right):
        left, right = self._vector(left), self._vector(right)
        if len(left) != len(right):
            raise ValueError("add requires equal vector lengths")
        self._capacity(len(left), 2 * len(left))
        nodes = tuple(Node("affine", ((a, Fraction(1)), (b, Fraction(1))))
                      for a, b in zip(left, right))
        return self._append(nodes, 2 * len(nodes))

    def concat(self, *vectors):
        self._open()
        sources = []
        for vector in vectors:
            nodes = self._vector(vector)
            if len(sources) + len(nodes) > MAX_VECTOR_HANDLES:
                raise ValueError("vector handle limit exceeded")
            sources.extend(nodes)
        return Vector(self._owner, tuple(sources))

    def constrain(self, vector, relation, rhs):
        sources = self._vector(vector)
        if relation not in ("eq", "le"):
            raise ValueError("relation must be eq or le")
        if len(self._predicates) + len(sources) > MAX_PREDICATES:
            raise ValueError("explicit predicate limit exceeded")
        rhs = _values(rhs, len(sources))
        predicates = tuple(Predicate(node, relation, value) for node, value in zip(sources, rhs))
        self._predicates.extend(predicates)

    def freeze(self, outputs):
        outputs = self._vector(outputs)
        element = Element(self.n_inputs, self.n_binary, tuple(self._nodes),
                          tuple(self._gates), tuple(self._predicates), outputs, self._owner)
        self._closed = True
        return element


def make_builder(n_inputs, n_binary=0, *, enabled=False):
    """Strict opt-in: disabled calls do not even validate dimensions."""
    return Builder(n_inputs, n_binary) if enabled is True else None


@dataclass(frozen=True, slots=True)
class Element:
    n_inputs: int
    n_binary: int
    nodes: tuple
    gates: tuple
    predicates: tuple
    outputs: tuple
    _owner: object

    @property
    def n_bin(self):
        return self.n_binary + len(self.gates)

    def evaluate(self, inputs, bits):
        inputs, bits = _values(inputs, self.n_inputs), _values(bits, self.n_bin)
        feasible = all(-1 <= x <= 1 for x in inputs) and all(b in (0, 1) for b in bits)
        values = []
        for node in self.nodes:
            if node.kind == "input":
                value = inputs[node.index]
            elif node.kind == "binary":
                value = bits[node.index]
            elif node.kind == "relu":
                value = _mul(bits[node.index], values[node.terms[0][0]])
            else:
                value = node.constant
                for source, coefficient in node.terms:
                    value = _add(value, _mul(coefficient, values[source]))
            values.append(value)
        for gate in self.gates:
            guard = _mul(_add(_mul(Fraction(2), bits[gate.bit]), Fraction(-1)),
                         values[gate.preactivation])
            feasible = feasible and guard >= 0
        for predicate in self.predicates:
            value = values[predicate.node]
            valid = value == predicate.rhs if predicate.relation == "eq" else value <= predicate.rhs
            feasible = feasible and valid
        return Evaluation(feasible, tuple(values), tuple(values[i] for i in self.outputs),
                          inputs, self._owner, bits)

    def lower(self):
        return _lower(self)


@dataclass(frozen=True, slots=True)
class Form:
    terms: tuple
    constant: Fraction = Fraction(0)

    def value(self, assignment):
        value = self.constant
        for index, coefficient in self.terms:
            value = _add(value, _mul(coefficient, assignment[index]))
        return value


@dataclass(frozen=True, slots=True)
class Row:
    terms: tuple
    rhs: Fraction
    relation: str


@dataclass(frozen=True, slots=True)
class GateMap:
    node: int
    preactivation: int
    bit: int
    value_var: int
    bit_var: int
    lower: Fraction
    upper: Fraction


def _form(parts=(), constant=Fraction(0)):
    terms = {}
    constant = _q(constant)
    for coefficient, form in parts:
        constant = _add(constant, _mul(coefficient, form.constant))
        for index, value in form.terms:
            terms[index] = _add(terms.get(index, Fraction(0)), _mul(coefficient, value))
    return Form(tuple((i, value) for i, value in sorted(terms.items()) if value), constant)


def _unit(index):
    return Form(((index, Fraction(1)),))


def _row(form, rhs=Fraction(0), relation="le"):
    return Row(form.terms, _add(rhs, _neg(form.constant)), relation)


@dataclass(frozen=True, slots=True)
class Lowered:
    n_inputs: int
    n_original_binary: int
    n_cont: int
    n_bin: int
    rows: tuple
    node_forms: tuple
    output_forms: tuple
    input_bounds: tuple
    continuous_bounds: tuple
    binary_ids: tuple
    gates: tuple
    node_bounds: tuple
    n_predicates: int
    affine_edges: int
    _owner: object

    @property
    def binary_columns(self):
        """Native variable columns, distinct from the retained bit identities."""
        return tuple(range(self.n_cont, self.n_cont + self.n_bin))

    @property
    def counts(self):
        row_nnz = sum(len(row.terms) for row in self.rows)
        node_nnz = sum(len(form.terms) for form in self.node_forms)
        return {"n_inputs": self.n_inputs, "n_original_binary": self.n_original_binary,
                "n_gates": len(self.gates), "n_bin": self.n_bin, "n_cont": self.n_cont,
                "n_nodes": len(self.node_forms), "n_predicates": self.n_predicates,
                "n_rows": len(self.rows), "row_nnz": row_nnz,
                "row_coefficients": row_nnz + len(self.rows),
                "node_form_nnz": node_nnz,
                "node_form_coefficients": node_nnz + len(self.node_forms),
                "output_form_nnz": sum(len(form.terms) for form in self.output_forms),
                "affine_edges": self.affine_edges}

    def assignment(self, evaluation, bits):
        if not isinstance(evaluation, Evaluation) or evaluation._owner is not self._owner:
            raise ValueError("evaluation belongs to another frozen input frame")
        bits = _values(bits, self.n_bin)
        if bits != evaluation._bits:
            raise ValueError("bits differ from the evaluated assignment")
        inputs = _values(evaluation.inputs, self.n_inputs)
        values = _values(evaluation.node_values, len(self.node_forms))
        assignment = inputs + tuple(values[gate.node] for gate in self.gates) + bits
        if any(form.value(assignment) != value for form, value in zip(self.node_forms, values)):
            raise ValueError("evaluation does not reconstruct all original nodes")
        if self.output_values(assignment) != evaluation.outputs:
            raise ValueError("evaluation outputs do not match")
        return assignment

    def satisfies(self, assignment):
        assignment = _values(assignment, self.n_cont + self.n_bin)
        feasible = all(lo <= value <= hi for value, (lo, hi) in
                       zip(assignment[:self.n_cont], self.continuous_bounds))
        feasible = feasible and all(b in (0, 1) for b in assignment[self.n_cont:])
        for row in self.rows:
            value = Form(row.terms).value(assignment)
            valid = value == row.rhs if row.relation == "eq" else value <= row.rhs
            feasible = feasible and valid
        return feasible

    def output_values(self, assignment):
        assignment = _values(assignment, self.n_cont + self.n_bin)
        return tuple(form.value(assignment) for form in self.output_forms)


def _lower(element):
    n_cont = element.n_inputs + len(element.gates)
    input_bounds = ((Fraction(-1), Fraction(1)),) * element.n_inputs
    continuous_bounds = list(input_bounds)
    forms, bounds, maps, rows = [], [], [], []
    for node_id, node in enumerate(element.nodes):
        if node.kind == "input":
            form, interval = _unit(node.index), input_bounds[node.index]
        elif node.kind == "binary":
            form, interval = _unit(n_cont + node.index), (Fraction(0), Fraction(1))
        elif node.kind == "affine":
            form = _form(((coefficient, forms[source]) for source, coefficient in node.terms),
                         node.constant)
            lower = upper = node.constant
            for source, coefficient in node.terms:
                lo, hi = bounds[source]
                if coefficient < 0:
                    lo, hi = hi, lo
                lower = _add(lower, _mul(coefficient, lo))
                upper = _add(upper, _mul(coefficient, hi))
            interval = lower, upper
        else:
            preactivation = node.terms[0][0]
            lower, upper = bounds[preactivation]
            value_var, bit_var = element.n_inputs + len(maps), n_cont + node.index
            form, bit, pre = _unit(value_var), _unit(bit_var), forms[preactivation]
            interval = max(Fraction(0), lower), max(Fraction(0), upper)
            continuous_bounds.append(interval)
            maps.append(GateMap(node_id, preactivation, node.index, value_var, bit_var, lower, upper))
            rows.extend((_row(_form(((Fraction(-1), form),))),
                         _row(_form(((Fraction(1), pre), (Fraction(-1), form)))),
                         _row(_form(((Fraction(1), form), (_neg(upper), bit)))),
                         _row(_form(((Fraction(1), form), (Fraction(-1), pre),
                                     (_neg(lower), bit))), _neg(lower))))
        forms.append(form)
        bounds.append(interval)
    for predicate in element.predicates:
        rows.append(_row(forms[predicate.node], predicate.rhs, predicate.relation))
    return Lowered(element.n_inputs, element.n_binary, n_cont, element.n_bin, tuple(rows),
                   tuple(forms), tuple(forms[i] for i in element.outputs), input_bounds,
                   tuple(continuous_bounds), tuple(range(element.n_bin)), tuple(maps),
                   tuple(bounds), len(element.predicates),
                   sum(len(node.terms) for node in element.nodes if node.kind == "affine"),
                   element._owner)
