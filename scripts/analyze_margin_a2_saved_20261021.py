"""Exact, saved-only diagnosis of ONE archived synthetic A2 source.

No solver, native library, new support query or online acceptance. This is a
source-specific algebra audit, not a general LP optimizer or network checker.
The accompanying document supplies the symbolic minimization/derivative proof.
"""
import base64
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "c80863cec99bb522c6d5461833abea1b79268f6ed22423bcceced4bfeb5ea355"
SOURCE_SHA = "2bcde55e513c38957a2b67269b4b8ed0c43ea464bc790a2ff15fa4556fd2afd2"
PROPERTY_SHA = "4dec36ad8c1b2fb1d0ab510193b57d9352cc1aac9e97b2c76e844a095fa18dbe"


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rational(value):
    # Fraction(float) interprets the stored binary64 exactly, not its decimal print.
    result = F(value)
    require(max(abs(result.numerator).bit_length(), result.denominator.bit_length()) <= 4096,
            "unchanged stored coefficient bit limit")
    return result


def row_sum(rows):
    out = {}
    for weight, row in rows:
        for j, v in row.items():
            out[j] = out.get(j, F(0)) + weight*v
    return {j: v for j, v in out.items() if v}


def csr(value):
    nr, nc = value["shape"]
    ptr, ids, data = value["indptr"], value["indices"], value["data"]
    require(len(ptr) == nr+1 and ptr[0] == 0 and ptr[-1] == len(ids) == len(data), "CSR shape")
    rows = []
    for i in range(nr):
        require(ptr[i] <= ptr[i+1], "CSR pointer order")
        indices = ids[ptr[i]:ptr[i+1]]
        require(indices == sorted(set(indices)) and all(0 <= j < nc for j in indices), "CSR indices")
        rows.append({j: rational(data[k]) for k, j in enumerate(indices, ptr[i])})
    return rows


def hz(value):
    nc, nb = value["Gc"]["shape"][1], value["Gb"]["shape"][1]
    result = dict(nc=nc, nb=nb, n=nc+nb)
    for key, ck, bk in (("G", "Gc", "Gb"), ("E", "Ac", "Ab"), ("A", "Auc", "Aub")):
        cr, br = csr(value[ck]), csr(value[bk])
        require(len(cr) == len(br), "matched HZ rows")
        result[key] = [c | {nc+j: v for j, v in b.items()} for c, b in zip(cr, br)]
    for key in ("c", "b", "ub"):
        result[key] = list(map(rational, value[key]))
    require(len(result["G"]) == len(result["c"]) and len(result["E"]) == len(result["b"])
            and len(result["A"]) == len(result["ub"]), "HZ right-hand sides")
    return result


def dot(row, point):
    return sum((v*point[j] for j, v in row.items()), F(0))


def project(parsed, weights, constant=F(0)):
    require(len(weights) == len(parsed["G"]), "all output rows included")
    return (constant+sum((w*c for w, c in zip(weights, parsed["c"])), F(0)),
            row_sum(zip(weights, parsed["G"])))


def affine_sum(terms):
    return (sum((w*a[0] for w, a in terms), F(0)), row_sum((w, a[1]) for w, a in terms))


def value(row, point):
    return row[0]+dot(row[1], point)


def feasible(parsed, point):
    require(len(point) == parsed["n"] and all(-1 <= v <= 1 for v in point), "all factor boxes")
    require(all(dot(r, point) == v for r, v in zip(parsed["E"], parsed["b"])), "all equalities")
    require(all(dot(r, point) <= v for r, v in zip(parsed["A"], parsed["ub"])), "all inequalities")


def main():
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.setitimer(signal.ITIMER_REAL, 60)
    raw = (ROOT/"docs/margin_a2_records_20261021.json").read_bytes()
    require(sha(raw) == ARCHIVE_SHA, "one immutable saved archive")
    archive = json.loads(raw)

    def read(name):
        file = archive["files"][name]
        item = archive["objects"][file["sha256"]]
        require(file["bytes"] == item["bytes"] <= 2*2**20, "bounded saved object")
        decoder = zlib.decompressobj()
        data = decoder.decompress(base64.b64decode(item["zlib_base64"], validate=True), item["bytes"]+1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                and len(data) == item["bytes"] and sha(data) == file["sha256"], "lossless saved bytes")
        return json.loads(data)

    common, zero, finite = (read(n) for n in ("common.json", "arm_zero.json", "arm_finite.json"))
    source, view = common["source"], common["view"]
    require(sha(compact(source)) == common["source_sha256"] == SOURCE_SHA
            and sha(compact(view)) == common["property_sha256"] == PROPERTY_SHA
            and view["source"] == SOURCE_SHA, "source and full property binding")
    p = hz(source["augmented"])
    require((p["nc"], p["nb"], len(p["G"])) == (3075, 4, 22), "fixed archived dimension")
    q = tuple(map(rational, view["q"]))
    require(q == tuple(F((-1)**j*(j+1), 16) for j in range(10)) and F(view["constant"]) == F(1,8),
            "full ten-class property, not a selected output")
    b = project(p, (F(0),)*10+q+(F(0), F(0)), F(1,8))
    d = project(p, q+tuple(-v for v in q)+(F(0), F(0)))
    m = project(p, (F(0),)*20+(F(1), F(-1)))
    for key, row in (("b", b), ("d", d), ("m", m)):
        encoded = dict(constant=str(row[0]), entries=[[j, str(v)] for j, v in sorted(row[1].items())])
        require(encoded == view["rows"][key], "independent original-row projection "+key)

    # These exact rows justify the eliminations and redundancies in the proof.
    require(p["E"] == [{1:F(1)}, {0:F(1)}, {3072:F(1),3076:F(-1,4)},
                       {3074:F(1),3078:F(-1,4)}] and p["b"] == [0]*4, "four original equalities")
    require(p["A"] == [{2:F(1)}, {0:F(1),3075:F(1,4)},
                       {3072:F(-1),3076:F(1,2)}, {3074:F(-1),3078:F(1,2)}]
            and p["ub"] == [F(3,4),F(1,2),F(3,4),F(3,4)], "four original inequalities")

    def reduce(row):
        subs = {0:{}, 1:{}, 3072:{3076:F(1,4)}, 3074:{3078:F(1,4)}}
        return row[0], row_sum((v, subs.get(j, {j:F(1)})) for j,v in row[1].items())

    nuisance = {j:F((-1)**j*(j+1)**2,512) for j in range(3,10)}
    require(reduce(b) == (F(-11,32), dict(nuisance) | {2:F(9,512),3075:F(-55,1024),3078:F(-165,1024)})
            and reduce(d) == (F(5,128), {3076:F(165,1024),3078:F(165,1024)})
            and reduce(m) == (F(1,2), {2:F(3,8),3075:F(3,16)}), "complete reduced objective")
    require(sum(map(abs, nuisance.values()), F(0)) == F(371,512), "independent nuisance support")
    lo, star, hi = F(-1,16), F(5,16), F(31,32)

    def path(mu):
        z = [F(0)]*p["n"]
        for j, v in nuisance.items():
            z[j] = -1 if v > 0 else 1
        z[2], z[3075] = ((F(-1),(16*mu-2)/3) if mu <= star else ((16*mu-11)/6,F(1)))
        z[3072],z[3074],z[3076],z[3077],z[3078] = F(-1,4),F(1,4),F(-1),F(1),F(1)
        return z

    def psi(mu, t):
        return F(-333,256)+5*t/128+(F(55,192)*(star-mu) if mu <= star else F(3,64)*(mu-star))

    # Each path segment is affine; endpoints establish all its linear constraints.
    # t dependence is affine too. The document proves global minimization, rather
    # than inferring it from these endpoint checks.
    for mu in (lo, star, hi):
        point = path(mu)
        feasible(p, point)
        require(value(m, point) == mu, "conditional path margin")
        for t in (F(0),F(1)):
            require(value(b,point)+t*value(d,point) == psi(mu,t), "attaining conditional affine path")
    witness = path(star)
    require(all(v in (-1,1) for v in witness[p["nc"]:]), "witness retains binary factor integrality")
    require([value(row,witness) for row in (b,d,m)] == [F(-333,256),F(5,128),star], "negative point projection")
    require(F(-333,256)+F(5,128) == F(-323,256) < 0, "negative feasible upper enclosure")

    # Check the same assignment in all six original source objects. This binds
    # the factor embedding, not an upstream trained network or native replay.
    mappings = {
        "entry": list(range(3072))+[3075], "router": list(range(3072))+[3075],
        "a": list(range(3073))+[3075,3076],
        "b": list(range(3072))+[3073,3074,3075,3077,3078],
        "input": list(range(3079)), "joint": list(range(3079)),
    }
    outputs, source_constraints = {}, {}
    require(set(source["base"]["sources"]) == set(mappings), "all six original sources")
    for name, mapping in mappings.items():
        original = hz(source["base"]["sources"][name])
        local = [witness[j] for j in mapping]
        feasible(original, local)
        require(all(v in (-1,1) for v in local[original["nc"]:]), "original binary factors")
        outputs[name] = [c+dot(r,local) for c,r in zip(original["c"],original["G"])]
        source_constraints[name] = [len(original["E"]),len(original["A"])]
    augmented_outputs = [c+dot(r,witness) for c,r in zip(p["c"],p["G"])]
    require(outputs["input"] == outputs["entry"], "same shared input outputs")
    require(outputs["a"] == augmented_outputs[:10] and outputs["b"] == augmented_outputs[10:20]
            and outputs["joint"] == augmented_outputs[:20], "same original expert outputs")
    require(source["base"]["context"]["pair"] == [0,2]
            and outputs["router"] == [F(1,4),F(-1,8),F(-1,16)]
            and augmented_outputs[20:] == [outputs["router"][0],outputs["router"][2]]
            and min(outputs["router"][0],outputs["router"][2]) > outputs["router"][1],
            "one legal synthetic pair at the negative point, not a route-change witness")

    # Independent exact dual reconstruction plus a matching feasible point gives
    # optimality of four DISTINCT supports. Six executions are established by
    # the prior full A2 event audit; this script checks their saved support rows.
    checked = []
    for arm in (zero, finite):
        for s in arm["record"]["supports"]:
            t, nu = F(s["t"]), F(s["nu"])
            require(F(-3,64) <= nu <= F(55,192) and 0 <= t <= 1, "analytic support slope interval")
            obj = affine_sum(((F(1),b),(t,d),(nu,m)))
            y,z = tuple(map(rational,s["y"])),tuple(map(rational,s["z"]))
            require(len(y) == len(p["A"]) and len(z) == len(p["E"]) and all(v <= 0 for v in y), "complete dual/sign")
            residual = row_sum([(F(1),obj[1])]+[(-v,r) for v,r in zip(y,p["A"])]+[(-v,r) for v,r in zip(z,p["E"])])
            bound = obj[0]+sum((v*r for v,r in zip(y,p["ub"])),F(0))+sum((v*r for v,r in zip(z,p["b"])),F(0))-sum(map(abs,residual.values()),F(0))
            require(bound == F(s["lower"]) == value(obj,witness) == F(-333,256)+5*t/128+nu*star,
                    "matching exact primal and dual global values")
            checked.append(dict(t=str(t),nu=str(nu),optimum=str(bound)))
    require(len(checked) == 6 and len({(v["t"],v["nu"]) for v in checked}) == 4, "six saved supports/four objectives")

    l,h = map(F,finite["record"]["plan"]["anchors"])
    require((l,h) == (F(27,64),F(53,64)), "unchanged anchor range")
    gains = []
    for fc,zc in zip(finite["record"]["cells"], finite["matched_zero"]["cells"]):
        x,y = F(fc["x"]),F(fc["y"])
        require(fc["gate"] == zc["gate"] and fc["x"] == zc["x"] and fc["y"] == zc["y"], "matched scalar cells")
        for mode,record,cell in (("finite",finite["record"],fc),("zero",finite["matched_zero"],zc)):
            a = []
            for t in (l,h):
                lines = [(F(s["lower"]),F(s["nu"])) for s in record["supports"] if F(s["t"]) == t]
                require(all(nu <= 0 for _,nu in lines), "common left cell minimizer")
                low = max(L-nu*x for L,nu in lines)
                require(low == F(-333,256)+5*t/128+(max(F(0),(x-star)/128) if mode == "finite" else 0), "exact hinge envelope")
                a.append(low)
            bound = min(((h-g)*a[0]+(g-l)*a[1])/(h-l) for g in map(F,cell["gate"]))
            require(bound == F(cell["lower"]), "saved scalar cell reconstruction")
        gains.append(F(fc["lower"])-F(zc["lower"]))
    require(gains == [0,0,F(3,2048),F(19,4096)], "non-bottleneck-only gains")
    sigma_lower,sigma_upper = F(1,2),F(1,2)+star/4
    bracket = [F(-333,256)+5*t/128 for t in (sigma_lower,sigma_upper)]
    require(bracket == [F(-41,32),F(-10471,8192)], "true property minimum enclosure")
    require(F(-55,192)+F(5,512) == F(-425,1536) < 0 and F(3,64) > 0, "monotone conditional branches")
    scalar_limit = F(5,128)*F(5,16)**3/48
    require(scalar_limit == F(625,25165824) < F(1,1024), "shared scalar-only improvement cap")
    require(not any(n in sys.modules for n in ("torch","numpy","scipy","act")), "saved standard-library analysis only")
    answer = dict(schema="A2_SAVED_CONDITIONAL_DIAGNOSIS_V1", status="PASS", offline_only=True,
        archive_sha256=ARCHIVE_SHA, source_sha256=SOURCE_SHA, property_sha256=PROPERTY_SHA,
        reachable_margin=[str(lo),str(hi)], conditional_break=str(star),
        exact_global_supports=checked, witness={str(j):str(v) for j,v in enumerate(witness) if v},
        witness_b_d_m=[str(value(r,witness)) for r in (b,d,m)],
        original_source_constraint_counts=source_constraints,
        synthetic_witness_router=list(map(str,outputs["router"])),
        true_given_P_minimum_enclosure=list(map(str,bracket)),
        cell_tilt_gains=list(map(str,gains)), scalar_only_improvement_upper=str(scalar_limit),
        decision="KEEP_A2_STOP_NO_NEW_EXECUTION_BASIS", new_solver_calls=0,
        full_model_UNSAFE=False, network_lowering_checked=False, old_online_status_changed=False,
        elapsed_seconds=time.monotonic()-start)
    print(json.dumps(answer, sort_keys=True))


if __name__ == "__main__":
    main()
