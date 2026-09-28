"""Read-only navigation/identity check; never starts experiments or deletes files.

Standard library only. --render prints the deterministic directory guide.
--workspace additionally classifies local MOE entries and checks local folders.
Neither a directory classification nor a successful hash check is a proof.
"""
import argparse
from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SPEC = 'docs/organization/layout.json'
CATALOG = 'docs/organization/DIRECTORY_CATALOG.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def directory_owners(spec):
    require(spec['schema'] == 1, 'unknown layout schema')
    owners = {}
    ids = set()
    for group in spec['groups']:
        require(group['id'] not in ids, 'duplicate group')
        ids.add(group['id'])
        for name in group['roots']:
            require(bool(name) and name not in ('.', '..') and
                    '/' not in name and '\\' not in name, 'invalid directory name')
            require(name not in owners, 'duplicate directory: ' + name)
            owners[name] = group['id']
    return owners


def workspace_category(name, spec):
    matches = [r for r in spec['workspace_rules']
               if any(fnmatch.fnmatchcase(name, p) for p in r['patterns'])]
    require(len(matches) == 1, 'unclassified or ambiguous workspace entry: ' + name)
    return matches[0]['category']


def local_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'path escapes root: ' + str(relative))
    return path


def check_history(root, history):
    raw = local_path(root, history['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == history['sha256'], 'handoff archive identity')
    require(len(raw.splitlines()) == history['lines'], 'handoff archive line count')


def check_bindings(root, source_file):
    sources = read_json(local_path(root, source_file))['sources']
    require(len(sources) == 619, 'unexpected frozen binding count')
    for relative, digest in sources.items():
        path = local_path(root, relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'frozen source changed: ' + relative)
    return len(sources)


def check_links(root, names, workspace=False):
    checked, deferred = 0, 0
    for name in names:
        document = local_path(root, name)
        for raw in re.findall(r'\[[^\]\n]+\]\(([^)\s]+)\)', document.read_text()):
            target = urlsplit(raw)
            if target.scheme or target.netloc or not target.path:
                continue
            path = (document.parent / unquote(target.path)).resolve()
            require(path.is_relative_to(root.parent.resolve()), 'link escapes workspace: ' + raw)
            if not path.is_relative_to(root.resolve()) and not workspace:
                deferred += 1
                continue
            require(path.exists(), 'broken link in ' + name + ': ' + raw)
            checked += 1
    return checked, deferred


def render_catalog(spec):
    directory_owners(spec)
    lines = ['# 工程目录分类（路径保持原位）', '',
             '由 `scripts/check_project_layout.py --render` 根据 [layout.json](layout.json) 生成。',
             '这是导航，不是目录迁移、删除清单或新的执行授权。', '',
             '## ACT 仓库顶层目录', '']
    for group in spec['groups']:
        lines.extend(['### ' + group['title'], '', group['policy'] + '。', ''])
        lines.extend('- [' + name + '/](../../' + name + ')' for name in group['roots'])
        lines.append('')
    lines.extend(['## MOE 工作区目录与资产', '',
                  '下表按名称分类，不扫描环境、原始数据或权重内容。',
                  '`--workspace` 会列出实际条目及数量，并拒绝未分类的新目录；不会清理任何条目。', '',
                  '| 类别 | 原位名称 / 模式 | 保留规则 |',
                  '|---|---|---|'])
    for rule in spec['workspace_rules']:
        lines.append('| ' + rule['category'] + ' | ' +
                     '、'.join('`' + p + '`' for p in rule['patterns']) +
                     ' | ' + rule['policy'] + ' |')
    lines.extend(['', '## 查实验不要只看目录名', '',
                  '- 当前状态以 [当前交接](../CODEX_HANDOFF.md) 指向的对应审计为准。',
                  '- `*_archive` 可能是审计代码包，不是可删除的旧结果。',
                  '- `test_*` 可能保留超时、错误和部分证据，不能按前缀直接删除。',
                  '- 作者仓库/环境/数据/权重与 ACT 不同仓，不应递归提交。',
                  '- 顶层根文件包括 Git 元数据、README、环境声明、许可证和工作区配置；',
                  '  分类表主要覆盖目录，保留这些根文件原位。', '',
                  '[返回工程总导航](../PROJECT_INDEX.md)', ''])
    return '\n'.join(lines)


def check(root=ROOT, workspace=False):
    root = root.resolve()
    spec = read_json(root / SPEC)
    owners = directory_owners(spec)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    tracked_roots = {p.split('/')[0] for p in tracked if '/' in p}
    require(tracked_roots == set(owners),
            'directory catalogue drift: missing=' + str(sorted(tracked_roots - set(owners))) +
            ', stale=' + str(sorted(set(owners) - tracked_roots)))
    for name in owners:
        require(local_path(root, name).is_dir(), 'missing directory: ' + name)
    require((root / CATALOG).read_text() == render_catalog(spec), 'generated catalogue drift')
    check_history(root, spec['history'])
    binding_count = check_bindings(root, spec['frozen_bindings'])
    links, deferred = check_links(root, spec['navigation_files'], workspace)
    editor = read_json(local_path(root, spec['workspace_file']))
    for folder in editor['folders']:
        path = local_path(root.parent, str(root.name + '/' + folder['path']))
        if workspace or path.is_relative_to(root):
            require(path.is_dir(), 'missing editor folder: ' + folder['path'])
    report = {'status': 'PASS', 'scope': 'navigation_and_recorded_identity_only',
              'tracked_directory_groups': len(spec['groups']),
              'tracked_directories': len(owners), 'links_checked': links,
              'local_links_deferred': deferred, 'frozen_sources_unchanged': binding_count,
              'history_lines_preserved': spec['history']['lines'],
              'experiments_started': 0, 'files_deleted': 0}
    if workspace:
        entries = {p.name: workspace_category(p.name, spec) for p in sorted(root.parent.iterdir())}
        report['workspace_entries'] = entries
        report['workspace_counts'] = dict(Counter(entries.values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--workspace', action='store_true')
    args = parser.parse_args()
    if args.render:
        print(render_catalog(read_json(ROOT / SPEC)), end='')
    else:
        print(json.dumps(check(workspace=args.workspace), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
