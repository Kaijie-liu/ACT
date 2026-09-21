"""Same full state; author's strict route-invariance sufficient obligations.

Joint static output includes all global-class margins (including zero blocks),
strict route dominance and nonzero selected-score. Not a dynamic export.
"""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn


def load_full(repo, checkpoint, digest):
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != digest:
        raise ValueError('full checkpoint binding')
    sys.path.insert(0, str(Path(repo) / 'src/Vision_Transformer_Pytorch'))
    from vision_transformer_moe import MetaMoE
    model = torch.load(checkpoint, map_location='cpu', weights_only=False).double().eval()
    if type(model) is not MetaMoE or model.meta_top_k != 1 or model.num_classes_list != [10, 10]:
        raise ValueError('unsupported author full object')
    return model


class InvariantObligations(nn.Module):
    def __init__(self, router, expert, widths, route, label, sign, margin):
        super().__init__()
        if sign not in [-1, 1] or not 0 <= route < len(widths) or not 0 <= label < sum(widths):
            raise ValueError('invalid obligation identity')
        self.router, self.expert, self.route, self.sign = router, expert, route, sign
        self.other_routes = [i for i in range(len(widths)) if i != route]
        total, width, offset = sum(widths), widths[route], sum(widths[:route])
        self.output_rows = nn.Linear(width, total-1, dtype=torch.float64)
        with torch.no_grad():
            self.output_rows.weight.zero_()
            self.output_rows.bias.fill_(-margin)
            for row, other in enumerate(k for k in range(total) if k != label):
                if offset <= label < offset+width:
                    self.output_rows.weight[row, label-offset] += 1
                if offset <= other < offset+width:
                    self.output_rows.weight[row, other-offset] -= 1
        self.output_rows.requires_grad_(False)
        self.eval()

    def forward(self, x):
        r = self.router(x)
        selected = r[:, self.route:self.route+1]
        output = self.expert(x)
        if isinstance(output, tuple):
            output = output[0]
        return torch.cat([selected*0, selected-r[:, self.other_routes],
                          selected*self.sign, self.output_rows(output)], dim=1)


def load_obligations(config, request_id, route, sign):
    cfg = json.loads(Path(config).read_text())
    request = next(r for r in cfg['requests'] if r['id'] == request_id)
    model = load_full(cfg['repo'], cfg['checkpoint'], cfg['files'][cfg['checkpoint']])
    return InvariantObligations(model.meta_gating_net, model.experts[route],
        model.num_classes_list, route, request['label'], sign, cfg['margin'])
