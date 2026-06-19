#!/usr/bin/env python3
"""vLLM-backed generation server matching tau2_adapter.GenServer's interface
(.generate(ids)->list, .stop()), so the existing rollout harnesses (Lean / SWE /
Terminal / tau2) can drive a big fp8 model (e.g. Qwen3-30B-A3B-FP8) for fast
rollouts with NO change to the episode code -- just pass this as `gen`.

Batching: rollout threads call generate(ids) concurrently and block; a single
worker thread coalesces up to max_batch requests within a short window and submits
them as ONE vLLM call (vLLM continuous-batches the prompt list internally).

LoRA: set_lora(path) hot-swaps the policy adapter for the next rollouts -- this is
how the RL loop syncs the freshly-updated LoRA into the generator each iter.
"""
import queue
import threading
import time

from vllm import LLM, SamplingParams
try:
    from vllm import TokensPrompt
except Exception:  # older/newer layout
    from vllm.inputs import TokensPrompt

from .rollout import TEMPLATE


class VLLMGenServer:
    def __init__(self, model_path, tok, max_new_tokens=400, temperature=1.0,
                 top_p=1.0, quantization=None, gpu_mem=0.85, max_model_len=8192,
                 max_batch=64, batch_window_s=0.04, enable_lora=False,
                 max_lora_rank=32, tensor_parallel_size=1, enforce_eager=False):
        self.tok = tok
        self.eot = tok.convert_tokens_to_ids(TEMPLATE.eot)
        self.max_new_tokens = max_new_tokens
        # prompts longer than this would make (prompt + max_tokens) exceed the
        # model's context window -> vLLM raises and kills the whole batch. We
        # instead end such episodes with an immediate EOT.
        self.max_prompt_len = max_model_len - max_new_tokens - 8
        # enforce_eager drops cudagraph memory -- needed when enable_lora + a big fp8
        # model would otherwise leave no room for KV cache (trades a little gen speed).
        kw = dict(model=model_path, gpu_memory_utilization=gpu_mem,
                  max_model_len=max_model_len, enable_lora=enable_lora,
                  enforce_eager=enforce_eager,
                  tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        if quantization:                      # "fp8" => online-quantize a bf16 ckpt
            kw["quantization"] = quantization  # (omit for a pre-quantized FP8 ckpt)
        if enable_lora:
            kw["max_lora_rank"] = max_lora_rank
        self.llm = LLM(**kw)
        self.sp = SamplingParams(temperature=temperature, top_p=top_p,
                                 max_tokens=max_new_tokens, stop_token_ids=[self.eot])
        self.window, self.max_batch = batch_window_s, max_batch
        self.q: queue.Queue = queue.Queue()
        self._stop = False
        self._lora = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_lora(self, path):
        """Point generation at a LoRA adapter dir (or None for the base model)."""
        if not path:
            self._lora = None
            return
        from vllm.lora.request import LoRARequest
        # bump id each swap so vLLM reloads the adapter weights
        i = (self._lora.lora_int_id + 1) if self._lora else 1
        self._lora = LoRARequest(f"policy{i}", i, path)

    def generate(self, ids: list) -> list:
        box, ev = {}, threading.Event()
        self.q.put((ids, box, ev))
        ev.wait()
        if "err" in box:
            raise box["err"]
        return box["out"]

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            try:
                first = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            t0 = time.time()
            while len(batch) < self.max_batch and (time.time() - t0) < self.window:
                try:
                    batch.append(self.q.get(timeout=self.window))
                except queue.Empty:
                    break
            # over-long prompts: end the episode immediately (return just EOT) so a
            # single overflowing rollout can't crash the whole batch / run.
            live = []
            for item in batch:
                ids, box, ev = item
                if len(ids) > self.max_prompt_len:
                    box["out"] = [self.eot]
                    ev.set()
                else:
                    live.append(item)
            if not live:
                continue
            try:
                prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _, _ in live]
                kw = {"use_tqdm": False}
                if self._lora is not None:
                    kw["lora_request"] = self._lora
                outs = self.llm.generate(prompts, self.sp, **kw)
                for (ids, box, ev), o in zip(live, outs):
                    g = list(o.outputs[0].token_ids)
                    if not g or g[-1] != self.eot:  # vLLM omits the stop token
                        g = g + [self.eot]
                    box["out"] = g
                    ev.set()
            except Exception as exc:
                for _, box, ev in live:
                    box["err"] = exc
                    ev.set()


if __name__ == "__main__":  # smoke: load + generate one prompt
    import sys
    from transformers import AutoTokenizer
    from .rollout import set_template
    mp = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-FP8"
    set_template(mp)
    tok = AutoTokenizer.from_pretrained(mp)
    q = "Action: tactic"  # any
    gs = VLLMGenServer(mp, tok, max_new_tokens=64, gpu_mem=0.85)
    ids = tok("Say hi in one word.", return_tensors=None)["input_ids"]
    out = gs.generate(ids)
    print("GEN:", repr(tok.decode(out)))
    gs.stop()
    print("VLLM SMOKE OK")
