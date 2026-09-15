"""Relocatable, stdlib-only reader for content-addressed request evidence.

This transport does not establish network-to-HZ correctness. The bundled
mathematical checker explicitly retains its original trusted assumptions.
"""
import hashlib
import json
from pathlib import Path
import zipfile


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class Store:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        names = self.zip.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive member')
        self.cache = {}

    def get(self, sha):
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid content reference')
        if sha not in self.cache:
            data = self.zip.read(sha)
            if digest(data) != sha:
                raise ValueError('content hash mismatch')
            self.cache[sha] = strict_json(data)
        return self.cache[sha]

    def decode(self, value):
        if isinstance(value, dict):
            if set(value) == {'$array'}:
                result = self.get(value['$array'])
                if not isinstance(result, list):
                    raise ValueError('array expected')
                return result
            return {k: self.decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        return value


def verify(directory, expected_bundle, expected_statement):
    root = Path(directory).resolve()
    raw = (root / 'bundle.json').read_bytes()
    if digest(raw) != expected_bundle:
        raise ValueError('bundle identity mismatch')
    meta = strict_json(raw)
    if meta['schema'] != 'PORTABLE_PRE_F0_REQUEST_V1':
        raise ValueError('unsupported bundle')
    for name, sha in meta['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path.read_bytes()) != sha:
            raise ValueError('bundle file identity/path mismatch')
    statement = meta['statement']
    if digest(compact(statement)) != expected_statement:
        raise ValueError('request statement changed')
    from portable_request import aggregate
    from act.back_end.solver.lp_certificate import identity
    store = Store(root / 'evidence.zip')
    def load(ref):
        name = ref['file']
        if Path(name).name != name:
            raise ValueError('nonlocal logical reference')
        item = meta['logical_files'][name]
        obj = store.decode(store.get(item['root']))
        if digest(original_bytes(obj)) != ref['sha256'] or ref['sha256'] != item['original_sha256']:
            raise ValueError('logical source identity mismatch')
        return obj
    manifest = load(meta['manifest'])
    if manifest['request'] != statement['request'] or manifest['routes'] != statement['routes']:
        raise ValueError('wrong theorem or route coverage')
    if {k: manifest[k] for k in ('common_facts', 'joint_source', 'router_source')} != statement['sources']:
        raise ValueError('trusted source binding changed')
    wrapper = load(manifest['common_facts'])
    if identity(wrapper['payload']) != wrapper['payload_sha256']:
        raise ValueError('snapshot identity mismatch')
    r = statement['request']
    job = {'parent_request_sha256': r['parent_request_sha256'],
           'case': {'expected_pairs': statement['routes']['feasible']},
           'parent_request': {'epsilon': r['epsilon'], 'subject': {'model_state': r['model_state']},
                              'sample': {'dataset_index': r['dataset_index'], 'label': r['clean_prediction'],
                                         **{k: r[k] for k in ('center', 'lower', 'upper')}}}}
    result = aggregate(manifest, wrapper['payload'], job, load)
    if result != meta['expected_result']:
        raise ValueError('result differs from archived request proof')
    return result
