"""Separate, unfused eval-BN compatibility path; not source equivalence proof."""
from pathlib import Path


def export_unfolded(model, example, destination):
    import torch
    import onnx
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('eval-only export')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and
           (not m.track_running_stats or m.running_mean is None or m.running_var is None)
           for m in model.modules()):
        raise ValueError('batch-dependent BN unsupported')
    expected = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if not expected:
        raise ValueError('this control requires retained BN')
    torch.onnx.export(model, example, str(destination), input_names=['input'],
                      output_names=['output'], opset_version=12, do_constant_folding=False,
                      training=torch.onnx.TrainingMode.EVAL)
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph)
    actual = sum(node.op_type == 'BatchNormalization' for node in graph.graph.node)
    if actual != expected:
        raise ValueError('BN was lost during export')
    return {'source_bn_count': expected, 'onnx_bn_count': actual,
            'node_types': [node.op_type for node in graph.graph.node],
            'constant_folding': False, 'onnx_simplification': False}


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(Path(path)), sess_options=options,
                               providers=['CPUExecutionProvider'])
