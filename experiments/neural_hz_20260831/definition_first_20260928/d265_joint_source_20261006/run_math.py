"""One frozen D265 joint-source component; complete mathematical stage."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d265_joint_source_20261006_v1'
PRIOR = EXP / 'results/d264_joint_support_kernel_20261006_v1'
D261_PRIOR = EXP / 'results/d261_mask_matched_reference_20261006_v1'
BASE = HERE.parent / 'd264_joint_support_kernel_20261006'
D261_BASE = HERE.parent / 'd261_mask_matched_reference_20261006'
DEPENDENCIES = HERE.parent / 'd214_parametric_relation_consumption_20261005/run_math.py'
THEORY = HERE.parent / 'd252_quantified_guard_capacity_20261006'
JOINT_THEORY = HERE.parent / 'd263_joint_residual_capacity_20261006'
TRANSPORT_THEORY = HERE.parent / 'd256_zero_readout_predicate_transport_20261006'
SOURCE_BASE = HERE.parent / 'd241_residual_source_binding_20261005'
SOURCE_RUN = EXP / 'results/d241_residual_source_binding_20261005_v1'
SOURCE_AUTHORITY = HERE.parent / 'd243_operator_source_relation_20261005/run_math.py'
OLD = HERE.parent / 'd130_import_isolation_20261002'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd265_joint_source_v1'
PLUGIN = ('experiments.neural_hz_20260831.definition_first_20260928.'
          'd265_joint_source_20261006.collection_contract')
FILES = ('CONTRACT.md', 'PREREG.md', 'source_arithmetic.py', 'joint_pair.py',
         'jp_certificate.py', 'source_observer.py', 'test_joint_source.py',
         'run_math.py', 'collection_contract.py', 'run_audit.py')
NAMES = (
    'test_01_grouped_source_and_bias',
    'test_02_fixed_gram_and_residual',
    'test_03_jp_certificate_controls',
    'test_04_default_off_budget_and_summary',
)
NEW_RECORD_FILES = ('summary.json',)
ANCHORS = {
    JOINT_THEORY / 'THEORY.md': '167f0d24d5ffec103245170f6db9de326df57df3c6ad90d97d63f57e3d2769be',
    JOINT_THEORY / 'CONTROL.md': 'b5770eeab798cf7bfdd059d66ebcac151e9f4de71468937d72f723dc49376a9e',
    JOINT_THEORY / 'SOURCE_PROGRAM.md': 'c8849826b3b2f05f42529068f256677bc6886dcca579ec87cda6d4a6d7aa65e7',
    JOINT_THEORY / 'RESEARCH_RECORD.md': '4b5b42928569b13df3ee096538feeab75325a332ca92553a7c5551a9b7a8562c',
    JOINT_THEORY / 'REVIEW.md': '0f97a732015594ffad2fb33d29dc427b3c827a1190b2b455398785394cd74ad5',
    JOINT_THEORY / 'ARCHIVE.sha256': 'c6d8f80e171fc88303aabc6d392a33bca8c934f48c9dfaf92f19b93b75ea2e97',
    HERE.parent / 'd253_quantified_native_component_20261006/freeze.json': '9ed1df8d814383de358efa05a9eb6bcbc862a4b77e67112ff069cdd71e0317ed',
    HERE.parent / 'd253_quantified_native_component_20261006/ARCHIVE.sha256': '730a4c2be3fa1e5a611d18a4bbaabc50fc8ab9bedc87070463b91423d888a1ca',
    EXP / 'results/d253_quantified_native_component_20261006_v1/exit.json': 'd3e44ae1463bcc921964560e2c1dbe748f81102093e77d3ddd4a2d1ee2cd5285',
    THEORY / 'ARCHIVE.sha256': 'f6e5a0c8a3341a342b2c7a898477ccc370be2471953ac976165888d729f8c198',
    THEORY / 'THEORY.md': '58a809e63dd1153bcf3055a18820dc27e25cc9be506e695e8d41c152b907fd1d',
    THEORY / 'CONTROL.md': '1970f7ac468119e541e1b17bbb9934bb84550a92d49e38f4d469fedf05434e1a',
    THEORY / 'CLIP_BOUNDARY.md': 'a158f8c12f9b11aac2a02f309d63120d89f10bed3e9df7e47600c9177a6c8764',
    THEORY / 'RESEARCH_RECORD.md': 'f65c606efce5349ca226a9d64d7498e00c0ed7ffb90c9dc42d024eee25286823',
    TRANSPORT_THEORY / 'THEORY.md': 'd3e6a14ac2b616e78c3f3ca8ef664f08c181b01389aac02f207eec9a7dc36148',
    TRANSPORT_THEORY / 'SOURCE_AUDIT.md': 'ae5219e5c32424d9737cc51769425cef510352f33c1b6591e396beffc2c0c579',
    TRANSPORT_THEORY / 'RESEARCH_RECORD.md': '23f84cf5e54cf2bd7f279614e1a455114e242bef0b5b59b37d51d4f1c74565b6',
    TRANSPORT_THEORY / 'ARCHIVE.sha256': '43ef0b3c06b4b26a72e1d293423e3369cf8eea2445a659bdeac955e60fd53486',
    OLD / 'run_math.py': 'cf7f9f1b767468430bcc3dfac861105f88625e74a4d1a7f0e6beb4f4744bc33d',
    DEPENDENCIES: 'f0f25fe16df7392ed8f407e612af3e51e965177549a832bf4daab01bc4041f42',
    SOURCE_AUTHORITY: 'd29050101ea55f9ccd314d2ba3dfad73bc581cf7b0365498b0b86628b9bbf8a0',
    (EXP / 'results/d257_zero_predicate_materialization_20261006_v1') / 'preregistered.json': 'db98e5e0485b35908628ed762ce54ebfd8e6642ad136debdba45f38c2961e8a3',
    (EXP / 'results/d257_zero_predicate_materialization_20261006_v1') / 'inventory.json': 'ba2b81af4c435e17d203415632d0d9a781f09bb7ffbb53491ae3133e242b0bb4',
    (EXP / 'results/d257_zero_predicate_materialization_20261006_v1') / 'exit.json': '2976ea1691830ea9b4db60e1bbe0f5aebe42a3539c8b036d6d09640ef03176c8',
    (EXP / 'results/d257_zero_predicate_materialization_20261006_v1') / 'summary.json': '2e228aa3a34210a34676631fd5ccb78899a81ca2aa61eec2018f59e4f70005d7',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'CONTRACT.md': 'c96a8757d7669633bfd178a14257580028e88661ec0ba27ef9d28c2c5d08573c',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'PREREG.md': 'eab5cfee52edfc574cfbeb58815cf7f7b923c289a19a466d6019a15493a6d3f0',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'predicate_transport.py': '9d9770e5117b6f7b96df7cbdf3627b9431aff7b67aded072df8c67ab0f8d28c8',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'test_transport.py': '2e2a126004eece3e538dc0488a659edf83bf801293d0882a254f77ce05a7ab54',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'run_math.py': 'ab95f190d71b30e0c57f33425ebae8bfa284052b99bf9fdf068f6c47bc154b12',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'collection_contract.py': '5d592c6b5d0fb5ef26d305b37892a13500564510c65c2cc446c2cd64b85fbc4e',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'RESULTS.md': 'b42b5d415564f5892ce771db3beed31eb0757d71d092e9e783f0274ce1014507',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'RESEARCH_RECORD.md': 'dd19e6936b82a52f905068ba04a85da1e315d89f9572d494cb0a0353503b0aa8',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'freeze.json': '2bbaa8c72354bc000429c340a90d251c7c9caec52a60963a6139af7571ab2af6',
    (HERE.parent / 'd257_zero_predicate_materialization_20261006') / 'ARCHIVE.sha256': 'cc15dd673e42b939a776264729ff788204f9b8752d4b649d3d266452038a9315',
    HERE.parent / 'd258_marginal_payment_redundancy_20261006/THEORY.md': '680706982423fcf9b2a6e1046516dbf83da29259438f8f76f14b532af5b77b7e',
    HERE.parent / 'd258_marginal_payment_redundancy_20261006/ARCHIVE.sha256': 'be79abfab2d50dd70439d9563d2c84f40c83573d8dc4ca584eb85f44831e7f31',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'CONTRACT.md': 'a77d3edbf4c7c3ae6aa184fe5ec974ae8fa58613509bc16b1e9720e97e028971',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'PREREG.md': 'a208d14e60c582e7585f03f60046ec63e0ed414c87a39c295d742b41f434e93c',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'source_bounds.py': '8a5ede20cf8c6e273beb55b6bc275095001bf581c6303a37c14fd6eb8031cf71',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'source_observer.py': '932e052f0563f5d93700156be87ada1e287e87753cbaa4120f45681e1e1a5d5b',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'test_source_bounds.py': 'cc10f854e2bd46c76ed64e0d28c6e972bab670e9b0a37554e4c5f26a5f95f109',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'run_math.py': '8e7e92be7d604020bbec2f806d808989dfb8bdfe4b6e733b00c844fa8c602583',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'collection_contract.py': 'a45dbd5b7bbc5f48bb1c372bbf6ba44e0a8e6e557421d644adebdb18936e6ee7',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'run_audit.py': 'c234b82960aeb0711e52fd3e5dba62f81cda481fdd381a72081c9b51942143e7',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'RESULTS.md': '5ffb872193c90e5471a9b6b76c6e34c2235b4c3e9891193629d809c6d88ee962',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'RESEARCH_RECORD.md': '23c45467481576f245a2a7f47bd4e9697c05272ab8540a347aab038b4b012b56',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'freeze.json': 'a8b75961a67e8a5d3873d796c9bd11ce25b3b9fb97fbf7a17cf24e44c756ba8d',
    (HERE.parent / 'd259_shared_difference_source_20261006') / 'ARCHIVE.sha256': '4af4c3a624d987f81e8e3b4a48b937bc7bc8d3346e6790d4d75ee73662858e48',
    (EXP / 'results/d259_shared_difference_source_20261006_v1') / 'preregistered.json': '2ba27415c835a458a885c7057e7222cca4b5bb059daf89a6900d942f8c7ba0ce',
    (EXP / 'results/d259_shared_difference_source_20261006_v1') / 'inventory.json': 'fe18da4b66af58a8e081d7eca320f2ed16c023294711b0a06bc952005e9b6760',
    (EXP / 'results/d259_shared_difference_source_20261006_v1') / 'exit.json': 'c112dc6333089afca2418eb5d87a84723b31f9f833b302be8f209f06bf791590',
    (EXP / 'results/d259_shared_difference_source_20261006_v1') / 'summary.json': 'a0fff566f175d5d116255712d7c4a576268389c85db3f0d05c9ee02337ccd870',
    HERE.parent / 'd260_mask_matched_reference_20261006/freeze.json': '02558b6c3b3f7fdcfb410e495e5b458323367b9f03332176a9ef1faf3935cc81',
    HERE.parent / 'd260_mask_matched_reference_20261006/ARCHIVE.sha256': '51c01d0c43afe652553f4e1ace3506598b9d26617308f752f6e287f3a7f99484',
    EXP / 'results/d260_mask_matched_reference_20261006_v1/exit.json': '985049c7a386be8312f261bc0858c81c952ed7174984f1b46a518bd7b32b8973',
    D261_BASE / 'CONTRACT.md': 'ca6a6c52555b2b0285d49175ce2fdbfc5bdc610b4862063fec06d451e2929e5d',
    D261_BASE / 'PREREG.md': 'abf87291242eaaa1eb1897e5da5e63f5259e4e3231dddd79a0567e2a92f9a754',
    D261_BASE / 'source_bounds.py': '8ffd0e6e74d912e873bf517064049751f7c40c07df7eaee455f014b24c482f46',
    D261_BASE / 'source_observer.py': 'f2c8e9d48fe785362a78ad8d80eddb77e5b4ebff13c11c238e89ab75c288d009',
    D261_BASE / 'test_source_bounds.py': 'bf54ec0c30490f3a31ca6d80df2a66fd6a442635096fbada57050dd500a70f19',
    D261_BASE / 'run_math.py': 'ee0e236931ed506e6e6027fea0fbacc1225f70276e48d1dc97a7df4a8f05a513',
    D261_BASE / 'collection_contract.py': '330d764d350cde6370634f03fec542274b6208fd366b081b3a504fc6ce4afb81',
    D261_BASE / 'run_audit.py': '506d773afc14a5beb0797f0034a14dae4d7f1e3f8653fe7f52d0a49522755c3a',
    D261_BASE / 'freeze.json': 'f144af5315d4b4f4595844dd5712e133c106b8df0cb1d14a10aa7d2133a5f360',
    D261_BASE / 'ARCHIVE.sha256': 'f717c454fd46434519f9754aacd2ce433d76c9073598059e68d1a4ce2cdaed5e',
    D261_BASE / 'RESULTS.md': '5f1553c3f121ba8a2634f64afe3f4b98ad8e322980d7830cc0531056ffb5b287',
    D261_BASE / 'RESEARCH_RECORD.md': '247b7d6e2cdbc3443bd9c91fcab0c862f45f8948c5d2a7f4118db965ed60bc14',
    D261_PRIOR / 'preregistered.json': '10d58fa44ddab43ece5a39edb561aceab409b832c5ff428aa99cb934a7a5e8f3',
    D261_PRIOR / 'inventory.json': '2bb2868e440fee25c88baa66f6e334cf163455d3f0b4bef6b699718bb290e657',
    D261_PRIOR / 'exit.json': '96ebb901ab64e67c1fbad3165ecbfddfe43dbf44b2f3539ecdd306db643786a1',
    D261_PRIOR / 'summary.json': '21a4ef693595d0e0e78003ec4a43ed66581396af942d0e46fa62d075aba0c760',
    BASE / 'CONTRACT.md': '0b24371692662df9684e9abfda436b6ed199bce5dcc08a59c8719a5dff891dde',
    BASE / 'PREREG.md': '0b2a55d93337dec659e887c657bb61969540af547b66eeda6af62a5bd84422bc',
    BASE / 'joint_support.py': '995a9d17fb1f2d2b67de1c594a35edfaf1d8e0d062006f589ec746938d6441bf',
    BASE / 'test_joint_support.py': 'af0398f1036d515bc1828814f4192e5f80aad0019374af64652021db7e7050f3',
    BASE / 'run_math.py': '560e36937adb1d2c19bfea3fda0e06375f28a31fbaee43c89c04ac65d95737d3',
    BASE / 'collection_contract.py': '20671679d90e5fc4cb6c03fb7df02c4f4c44914e9d38495919e4da7b3552927b',
    BASE / 'freeze.json': 'a95d8cbb27588d04cb553e6505fbc9f0d51a876a0d65f1e70e2233aeecc651aa',
    BASE / 'RESULTS.md': '9618faacd8fab5a7756f85b1bb20d5a9f7c13a45314fd1e5a005ba63a68c4869',
    BASE / 'RESEARCH_RECORD.md': '1f0d4ac4a28d83a3ae1707cefd86709a5636abba61611cebabf121172b3e7e20',
    BASE / 'REVIEW.md': '33eba9d50194ab963611c5e88e83840cd0cac1bdb35af0dd1adbd6fdaa5e7b5c',
    BASE / 'ARCHIVE.sha256': 'befa5d3973f5477e4098b87eca8361902fcb06353bcf15eaec11a6851a5740e5',
    PRIOR / 'preregistered.json': '53fc507ce23bac59c751eb8d67462acd37ff215a49c4c1251ad05d6a115442b8',
    PRIOR / 'inventory.json': '4841099297986ebb4259428da56688986395fa32548810dab5717ba7fca15cc7',
    PRIOR / 'exit.json': '2575b75d5eff761cfef1b9a6acc642b5b4e68f7acf66d2dda29c27a458fa25c3',
    PRIOR / 'summary.json': '6f8390e637270d421a327519928b4dd32437dd941072cebe153ff059e0444f2b',
}
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
TRUE_FLAGS = ('mathematical_stage_only', 'fixed_component_lp_controls_registered',
              'new_component_solver_free', 'source_audit_stage_registered')
FALSE_FLAGS = ('worker_stage_registered', 'worker_launched', 'source_component_qualified',
    'source_census_completed', 'source_census_qualified', 'actual_model_binding_qualified',
    'actual_phase_column_binding_verified', 'native_HZ_admitted', 'gpu_computation_completed',
    'complete_physical_qualification', 'candidate_physical_gate_evaluated',
    'production_snapshot_imported', 'solver_rescue_registered', 'negative_audit_only',
    'new_set_class', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed', 'domain_definition_changed',
    'native_runtime_installation_qualified', 'online_lifecycle_qualified',
    'native_mathematical_transport_passed', 'quantified_native_transport_passed',
    'rebase_native_transport_passed', 'zero_predicate_materialization_passed',
    'shared_difference_bounds_math_passed', 'mask_matched_reference_math_passed',
    'joint_support_kernel_math_passed')


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError('invalid JSON identity: ' + str(path))
    return json.loads(path.read_text())


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_checked(path, name, identities):
    if sha(path) != identities.get(str(path)):
        raise ValueError('unauthenticated helper: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    identities, inputs = {}, {}
    old = inherited = source_authority = dependency_helper = helper = gpu = production = closure = prior = None
    manifest_digest = None
    result = dict(schema=SCHEMA, new_tests=4, formal_gain=0, independent_e0_gain=0, new_benchmark_solves=0,
        component_tests_passed=False, mathematical_component_gate_passed=False,
        joint_source_math_passed=False,
        inventory_validated_before_execution=False)
    result.update({key: True for key in TRUE_FLAGS})
    result.update({key: False for key in FALSE_FLAGS})
    try:
        limits()
        if 0 not in os.sched_getaffinity(0):
            raise ValueError('required CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='', CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        sys.path.insert(0, str(ROOT))
        frozen = read(FREEZE)
        if (frozen.get('schema') != SCHEMA or frozen.get('required_tests') != 4289
                or frozen.get('required_test_files') != 232
                or frozen.get('new_test_names') != list(NAMES)
                or frozen.get('new_evidence_files') != list(NEW_RECORD_FILES)
                or any(frozen.get(key) is not True for key in TRUE_FLAGS)
                or any(frozen.get(key) is not False for key in
                       ('worker_stage_registered', 'solver_rescue_registered',
                        'negative_audit_only', 'new_set_class', 'domain_definition_changed',
                        'joint_source_math_passed', 'joint_support_kernel_math_passed',
                        'quantified_native_transport_passed',
                        'native_mathematical_transport_passed',
                        'native_runtime_installation_qualified', 'online_lifecycle_qualified',
                        'rebase_native_transport_passed', 'zero_predicate_materialization_passed'))
                or type(frozen.get('source_sha256')) is not dict
                or set(frozen['source_sha256']) != {str(HERE / name) for name in FILES}):
            raise ValueError('frozen D265 contract differs')
        identities.update(frozen['source_sha256'])
        identities.update({str(path): digest for path, digest in ANCHORS.items()})
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in identities.items():
            if sha(path) != digest:
                raise ValueError('frozen source or anchor differs: ' + path)
        old = load_checked(OLD / 'run_math.py', '_d265_authenticated_d130', identities)
        inherited = load_checked(BASE / 'run_math.py', '_d265_authenticated_d264', identities)
        dependency_helper = load_checked(DEPENDENCIES, '_d265_authenticated_dependencies', identities)
        prior, done, inventory = (read(PRIOR / name) for name in
                                  ('preregistered.json', 'exit.json', 'inventory.json'))
        prior_freeze = read(BASE / 'freeze.json')
        if (prior.get('schema') != 'd264_joint_support_kernel_v1'
                or prior.get('required_tests') != 4285 or prior.get('required_test_files') != 231
                or len(prior.get('source_sha256', {})) != 8092
                or len(prior.get('input_sha256', {})) != 14
                or prior.get('cpu_affinity') != [0]
                or any(prior.get(key) is not True for key in inherited.TRUE_FLAGS)
                or any(prior.get(key) is not False for key in inherited.FALSE_FLAGS)
                or any(done.get(key) is not True for key in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed',
                    'joint_support_kernel_math_passed'))
                or any(done.get(key) is not False for key in inherited.FALSE_FLAGS)
                or done.get('negative_audit_only') is not False
                or done.get('formal_gain') != 0 or done.get('new_benchmark_solves') != 0
                or done.get('tests_exit') != 0 or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 4285 or done.get('test_files') != 231
                or not 0 <= done.get('test_wall_s', 61) <= 60 or 'failure' in done
                or done.get('source_drift') != [] or done.get('input_drift') != []
                or done.get('provenance_drift') is not False
                or inventory.get('nodeids') != prior.get('expected_nodeids')
                or inventory.get('count') != 4285 or inventory.get('files') != 231
                or inventory.get('manifest_sha256') != ANCHORS[PRIOR / 'preregistered.json']
                or inventory.get('validated_before_execution') is not True
                or prior_freeze.get('source_sha256') != {
                    str(BASE / name): ANCHORS[BASE / name] for name in inherited.FILES}):
            raise ValueError('D264 complete mathematical receipt differs')
        for path, digest in prior['source_sha256'].items():
            old.merge_identity(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            path = PRIOR / name
            if path.resolve() != path or not path.is_relative_to(PRIOR):
                raise ValueError('historical artifact escapes original directory')
            old.merge_identity(identities, str(path), digest)
        source_authority = load_checked(SOURCE_AUTHORITY, '_d265_authenticated_d243_source_reference', identities)
        source_record = source_authority.source_reference(
            old, identities, inputs, read(source_authority.PRIOR / 'preregistered.json'))
        for path, digest in {**identities, **inputs}.items():
            if sha(path) != digest:
                raise ValueError('inherited identity drift: ' + path)
        if (Path(sys.executable).resolve() != old.PYTHON.resolve()
                or sha(Path(sys.executable).resolve()) != identities[str(old.PYTHON.resolve())]):
            raise ValueError('interpreter differs')
        closure = old.project_closure(identities, prior.get('project_import_closure'))
        helper = load_checked(old.HELPER, '_d265_authenticated_d015', identities)
        gpu = load_checked(old.GPU, '_d265_authenticated_d017_dependencies', identities)
        dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
        production = helper.provenance()
        if production != prior['provenance'] or list(os.sched_getaffinity(0)) != [0]:
            raise ValueError('production provenance or CPU affinity differs')
        tests, expected = list(prior['tests']), list(prior['expected_nodeids'])
        if (len(tests) != 231 or len(set(tests)) != 231
                or len(expected) != 4285 or len(set(expected)) != 4285):
            raise ValueError('inherited ordered population differs')
        test_path = HERE / 'test_joint_source.py'
        functions = [node for node in ast.parse(test_path.read_text()).body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if ([node.name for node in functions] != list(NAMES)
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                    or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                    or node.args.vararg or node.args.kwarg for node in functions)):
            raise ValueError('four registered plain test functions required')
        tests.append(str(test_path))
        expected.extend(str(test_path.relative_to(ROOT)) + '::' + name for name in NAMES)
        if (len(set(tests)) != 232 or len(set(expected)) != 4289
                or any(path not in identities for path in tests)
                or helper.drift(identities) or helper.drift(inputs)):
            raise ValueError('pre-execution identity or population drift')
        relocations = {}
        for key in ('inherited_component_evidence_relocation', 'inherited_D207_evidence_relocation',
                    'inherited_D208_evidence_relocation', 'inherited_D209_evidence_relocation',
                    'inherited_D214_evidence_relocation', 'inherited_D228_evidence_relocation',
                    'inherited_D229_evidence_relocation', 'inherited_D230_evidence_relocation',
                    'inherited_D231_evidence_relocation', 'inherited_D240_evidence_relocation',
                    'inherited_D243_evidence_relocation', 'inherited_D245_evidence_relocation',
                    'inherited_D249_evidence_relocation', 'inherited_D254_evidence_relocation',
                    'inherited_D255_evidence_relocation', 'inherited_D257_evidence_relocation',
                    'inherited_D259_evidence_relocation', 'inherited_D261_evidence_relocation'):
            relocation = dict(prior[key])
            relocation['relocated_run'] = str(RUN / Path(relocation['relocated_run']).name)
            relocations[key] = relocation
        relocations['inherited_D264_evidence_relocation'] = dict(
            module_path=str(BASE / 'test_joint_support.py'), source_sha256=ANCHORS[BASE / 'test_joint_support.py'],
            function_name='_record_file', function_firstlineno=56,
            original_run=str(PRIOR), relocated_run=str(RUN / 'inherited_d264_controls'),
            allowed_filenames=['summary.json'], mechanism='module_local_record_function_only')
        # Preserve all authenticated historical definitions/receipts, but never
        # inherit a qualification or the previous candidate's execution identity.
        manifest = dict(prior)
        manifest.update(schema=SCHEMA, source_sha256=identities, input_sha256=inputs,
            provenance=production, project_import_closure=closure, tests=tests,
            expected_nodeids=expected, required_tests=4289, required_test_files=232,
            inherited_tests=4285, inherited_test_files=231, new_test_files=1,
            new_test_names=list(NAMES), new_evidence_files=list(NEW_RECORD_FILES),
            inherited_test_population_unchanged=True,
            inherited_D264_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                inventory_sha256=ANCHORS[PRIOR / 'inventory.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'],
                mathematical_component_gate_passed=True, qualification_transferred=False),
            failed_D260_reference=dict(path=str(HERE.parent / 'd260_mask_matched_reference_20261006'),
                mathematical_component_gate_passed=False, population_inherited=False,
                qualification_transferred=False, reason='pre-execution anchor transcription failure'),
            preserved_D264_operator_definition=prior['operator_definition'],
            preserved_D264_candidate_semantic_definition=prior['candidate_semantic_definition'],
            last_successful_candidate_semantic_definition=prior['candidate_semantic_definition'],
            candidate_semantic_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                qualification_transferred=False),
            operator_definition=dict(path=str(HERE / 'CONTRACT.md'),
                sha256=frozen['source_sha256'][str(HERE / 'CONTRACT.md')],
                kind='complete_same_source_joint_residual_audit',
                domain_definition_changed=False, new_set_class=False),
            D241_source_reference=source_record,
            D263_joint_residual_theory_reference=dict(path=str(JOINT_THEORY),
                source_sha256={str(path): digest for path, digest in ANCHORS.items()
                               if path.is_relative_to(JOINT_THEORY)},
                qualification_transferred=False),
            D256_transport_theory_reference=dict(path=str(TRANSPORT_THEORY),
                source_sha256={str(path): digest for path, digest in ANCHORS.items()
                               if path.is_relative_to(TRANSPORT_THEORY)},
                qualification_transferred=False),
            D252_theorem_reference={str(path): digest for path, digest in ANCHORS.items()
                                   if path.is_relative_to(THEORY)},
            freeze_sha256=sha(FREEZE), collection_plugin=PLUGIN, pytest_import_mode='importlib',
            pytest_plugin_autoload=False, same_process_collection_gate=True,
            single_pytest_process=True, cpu_affinity=[0], address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, cuda_visible_devices='', formal_gain=0,
            independent_e0_gain=0, new_benchmark_solves=0,
            source_audit_stage_registered=True,
            joint_source_math_passed=False, **relocations)
        manifest.update({key: True for key in TRUE_FLAGS})
        manifest.update({key: False for key in FALSE_FLAGS})
        save('preregistered.json', manifest)
        manifest_digest = sha(RUN / 'preregistered.json')
        env['NEURAL_HZ_D265_MANIFEST_SHA256'] = manifest_digest
        env['NEURAL_HZ_ACTIVE_COMPONENT_RUN'] = str(RUN)
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
            '--tb=short', '-p', 'no:cacheprovider', '-p', PLUGIN,
            '--junitxml=' + str(RUN / 'tests.xml'), *tests]
        with (RUN / 'tests.log').open('x') as stream:
            test_started = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        result.update(tests_exit=process.returncode, test_wall_s=time.monotonic() - test_started,
                      tests_count=4289, test_files=232)
        test_started = None
        checked = read(RUN / 'inventory.json')
        if (checked.get('nodeids') != expected or checked.get('count') != 4289
                or checked.get('files') != 232 or checked.get('manifest_sha256') != manifest_digest
                or checked.get('validated_before_execution') is not True
                or any(checked.get(key) != value for key, value in relocations.items())
                or sha(RUN / 'preregistered.json') != manifest_digest):
            raise ValueError('pre-execution inventory contract differs')
        result['inventory_validated_before_execution'] = True
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (process.returncode != 0 or result['test_wall_s'] > 60 or actual != expected
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete mathematical test gate failed')
        if any((RUN / name).is_symlink() or not (RUN / name).is_file() for name in NEW_RECORD_FILES):
            raise ValueError('complete D265 mathematical evidence missing')
        result['component_tests_passed'] = True
        result['mathematical_component_gate_passed'] = True
        result['joint_source_math_passed'] = True
    except BaseException as exc:
        if test_started is not None:
            result['test_wall_s'] = time.monotonic() - test_started
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096], stage='mathematical')
    finally:
        try:
            if closure is not None and old.project_closure(identities, closure) != closure:
                raise ValueError('post-execution project closure differs')
            if helper is not None and gpu is not None and prior is not None:
                dependency_helper.dependencies(helper, gpu, identities, inputs, prior)
                if list(os.sched_getaffinity(0)) != prior['cpu_affinity']:
                    raise ValueError('post-execution CPU affinity differs')
            result['source_drift'] = [path for path, digest in identities.items() if sha(path) != digest]
            result['input_drift'] = [path for path, digest in inputs.items() if sha(path) != digest]
            result['provenance_drift'] = production is not None and helper.provenance() != production
            if (result['source_drift'] or result['input_drift'] or result['provenance_drift']
                    or (manifest_digest is not None and sha(RUN / 'preregistered.json') != manifest_digest)):
                raise ValueError('post-execution identity drift')
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        try:
            result['artifacts'] = {str(path.relative_to(RUN)): sha(path)
                                   for path in RUN.rglob('*') if path.is_file()}
        except BaseException as exc:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type=type(exc).__name__,
                reason='artifact sealing incomplete: ' + str(exc)[:4096]))
        try:
            if old is None:
                raise ValueError('authenticated telemetry helper unavailable')
            old.host_observations(result, rss0)
        except BaseException as exc:
            result['host_observations_within_caps'] = False
            result.setdefault('failure', dict(type=type(exc).__name__, reason=str(exc)[:4096]))
        if not result['host_observations_within_caps']:
            result['mathematical_component_gate_passed'] = False
            result.setdefault('failure', dict(type='MemoryError', reason='supervisor memory gate failed'))
        passed = result['mathematical_component_gate_passed'] and 'failure' not in result
        result['joint_source_math_passed'] = passed
        result.update(all_registered_stages_passed=passed, supervisor_exit=0 if passed else 1,
            wall_s=time.monotonic() - started,
            memory_scope='supervisor observed; pytest AS/CPU/time only; no GPU or full physical qualification')
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
