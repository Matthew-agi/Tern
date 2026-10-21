"""Streaming checkpoint names and coverage, independent of CUDA/vLLM imports."""


class PoolingWeights:
    def __init__(self, weights, head_names, exit_layer=None):
        if exit_layer is not None and (type(exit_layer) is not int or not 1 <= exit_layer <= 40):
            raise ValueError("Exit layer must be an integer from 1 to 40")
        self.weights = weights
        self.expected_head = {"pooler.head." + key for key in head_names}
        self.exit_layer = exit_layer
        self.seen_head = set()
        self.seen_output = False
        self.skipped_layers = set()

    def __iter__(self):
        for name, weight in self.weights:
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "model.", 1)
            elif name == "lm_head.weight":
                name = "pooler.output_weight"
            elif not name.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "pooler.")):
                continue
            if name.startswith("pooler.head."):
                if name in self.seen_head or name not in self.expected_head:
                    raise ValueError(f"Unexpected or duplicate head tensor: {name}")
                self.seen_head.add(name)
            if name == "pooler.output_weight":
                if self.seen_output:
                    raise ValueError("Duplicate output embedding")
                self.seen_output = True
            if name.startswith("model.layers."):
                layer = int(name.split(".")[2])
                if not 0 <= layer < 40:
                    raise ValueError("Unexpected checkpoint layer")
                if self.exit_layer is not None and layer >= self.exit_layer:
                    self.skipped_layers.add(layer)
                    continue
            yield name, weight

    def verify(self):
        if self.seen_head != self.expected_head or not self.seen_output:
            raise ValueError(f"Incomplete scoring checkpoint: missing head tensors {sorted(self.expected_head - self.seen_head)}, output embedding present={self.seen_output}")
        if self.exit_layer is not None and self.skipped_layers != set(range(self.exit_layer, 40)):
            raise ValueError(f"Incomplete frozen checkpoint tail: {self.skipped_layers}")
