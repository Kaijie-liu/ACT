"""Untrusted writer for request-local given-HZ pages; no accepting logic."""
from pathlib import Path

from scoped_source.hz_request_store import SCHEMA, POLICY, compact, digest, require, sha


class Writer:
    def __init__(self, root, request, experts, factors, *, work, page_rows=64):
        require(type(page_rows) is int and 1 <= page_rows <= POLICY['page_rows'], 'page row capacity')
        self.root = Path(root); self.root.mkdir(exist_ok=False, parents=True)
        self.work, self.page_rows = work, page_rows
        self.document = dict(schema=SCHEMA, request=sha(request), policy=dict(POLICY),
                             experts=experts, factors=factors, members=[], views=[], roles=None)
        self.members = {}; self.bytes = 0

    def rows(self, kind, rows):
        page = []; refs = []
        for row in rows:
            self.work.tick(); page.append(row)
            if len(page) == self.page_rows: refs.append(self._page(kind,page)); page = []
        if page: refs.append(self._page(kind,page))
        return refs

    def _page(self, kind, page):
        self.work.tick(); raw = compact(dict(kind=kind,rows=page)); h = digest(raw)
        require(len(raw) <= POLICY['member_bytes'], 'writer member capacity')
        if h not in self.members:
            require(len(self.members) < POLICY['members'], 'writer member count')
            self.bytes += len(raw); require(self.bytes <= POLICY['total_bytes'], 'writer total byte capacity')
            name = f'rows/{h}.json'; path = self.root/name; path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream: stream.write(raw)
            self.members[h] = dict(file=name,bytes=len(raw),sha256=h)
        self.work.tick(); return h

    def view(self, name, *, scope, guard=None, parent=None, frame=1, add_factors=(), output=(), eq=(), ine=()):
        self.document['views'].append(dict(name=name,scope=scope,guard=guard,parent=parent,frame=frame,
            add_factors=list(add_factors),output=list(output),eq=list(eq),ine=list(ine)))

    def finish(self, roles):
        self.work.tick(); self.document['roles'] = roles
        self.document['members'] = [self.members[h] for h in sorted(self.members)]
        raw = compact(self.document)
        require(len(raw) <= POLICY['header_bytes'] and len(raw)+self.bytes <= POLICY['total_bytes'], 'writer header/total capacity')
        with (self.root/'manifest.json').open('xb') as stream: stream.write(raw)
        self.work.tick(); return digest(raw)
