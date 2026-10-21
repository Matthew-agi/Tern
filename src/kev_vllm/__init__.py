def register():
    import os
    if os.environ.get('CLEF_EXPERT_ACT_QUANT_FUSION') == '1':
        from .expert_act_quant import install
        install()
    if os.environ.get('TERN_H100_ROWWISE_GEMM') == '1':
        from .dense_fp8 import install as install_dense
        install_dense()
    if (os.environ.get('CLEF_ACTIVE_EXPERTS', '8') != '8'
            or os.environ.get('CLEF_NORMALIZATION_REFERENCE', '8') != '8'):
        from .expert_reference_routing import install as install_reference_routing
        install_reference_routing()
    from vllm import ModelRegistry
    ModelRegistry.register_model("KevQwenForPooling", "kev_vllm.model:KevQwenForPooling")
    ModelRegistry.register_model("ClefQwenMoeForPooling", "kev_vllm.clef_model:ClefQwenMoeForPooling")
    # The stock pooling HTTP request does not expose PoolingParams.extra_kwargs.
    # Carry our schema spans through the allowed extra request fields while
    # leaving ordinary pooling requests unchanged.
    from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
    if not getattr(PoolingCompletionRequest, "_clef_metadata_patch", False):
        original = PoolingCompletionRequest.to_pooling_params
        def with_clef_metadata(self):
            params = original(self)
            extra = self.model_extra or {}
            if "clef_record" in extra:
                params.extra_kwargs = {"clef_record": extra["clef_record"]}
            return params
        PoolingCompletionRequest.to_pooling_params = with_clef_metadata
        PoolingCompletionRequest._clef_metadata_patch = True
    # The batched HTTP experiment supplies one independent schema per prompt.
    # Clone parameters so no request can overwrite another prompt's metadata.
    from vllm.entrypoints.pooling.base.serving import PoolingBaseServing
    if not getattr(PoolingBaseServing, '_clef_batch_metadata_patch', False):
        original_prepare = PoolingBaseServing._prepare_generators
        async def prepare_clef_batch(self, ctx):
            import copy
            schemas = (getattr(ctx.request, 'model_extra', None) or {}).get('clef_records')
            if schemas is not None:
                if (not isinstance(schemas, list) or ctx.engine_inputs is None
                        or len(schemas) != len(ctx.engine_inputs)
                        or any(not isinstance(s, dict) or not s.get('questions') for s in schemas)):
                    raise ValueError('Exactly one complete CLEF schema is required per input')
                for engine_input, schema in zip(ctx.engine_inputs, schemas):
                    params = copy.deepcopy(engine_input['params'])
                    params.extra_kwargs = {'clef_record': schema}
                    engine_input['params'] = params
            return await original_prepare(self, ctx)
        PoolingBaseServing._prepare_generators = prepare_clef_batch
        PoolingBaseServing._clef_batch_metadata_patch = True
