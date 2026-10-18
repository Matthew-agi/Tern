"""Opt-in bounded CUDA graphs for the tensor-only joint head, excluding encoding."""
from collections import OrderedDict
import torch


def replay_head(head, inputs, counts):
    from .clef_batched import forward_tensor
    if head.training or torch.is_grad_enabled() or not inputs[0].is_cuda:
        raise ValueError("Head capture requires CUDA inference")
    key = (tuple(counts), tuple((tuple(t.shape), t.dtype, str(t.device)) for t in inputs))
    cache = getattr(head, "_clef_head_graphs", None)
    if cache is None:
        cache = head._clef_head_graphs = OrderedDict()
        head._clef_head_graph_captures = head._clef_head_graph_replays = 0
    if key not in cache:
        while len(cache) >= head.clef_head_graph_entries:
            cache.popitem(last=False)
        stable = tuple(t.clone() for t in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward_tensor(head, *stable[:6], counts, *stable[6:])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward_tensor(head, *stable[:6], counts, *stable[6:])
        cache[key] = (stable, graph, output)
        head._clef_head_graph_captures += 1
    stable, graph, output = cache[key]
    cache.move_to_end(key)
    for target, source in zip(stable, inputs):
        target.copy_(source, non_blocking=True)
    graph.replay()
    head._clef_head_graph_replays += 1
    # Callers may retain logits across subsequent replays.
    return [tensor.clone() for tensor in output]
