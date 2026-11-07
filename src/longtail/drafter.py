"""Self + sibling n-gram drafter, loaded by vLLM's ``custom_class`` speculative method.

    speculative_config = {"method": "custom_class",
                          "model": "longtail.drafter.SiblingProposer",
                          "num_speculative_tokens": 8}

Needs vLLM's V1 model runner (VLLM_USE_V2_MODEL_RUNNER=0); the V2 runner only supports
built-in speculators. Configured through environment variables, which the engine-core
process inherits from the experiment process:

    LONGTAIL_DRAFT_SOURCES   self | sibling | both   (default both)
    LONGTAIL_DRAFT_GATE      draft only while <= this many sequences are decoding (0 = always)
    LONGTAIL_DRAFT_MIN_N / LONGTAIL_DRAFT_MAX_N   n-gram match lengths (default 2 / 4)

Siblings are the other samples of the same prompt. The rollout harness tags request ids
as ``g{group}-r{n}``; untagged requests just get self-drafting.

Per request we mirror its token ids (prompt + output) and keep, for each n, the latest
position of every n-gram that already has a continuation token. Drafting looks up the
request's last n tokens, longest n first, in its own index and every sibling's, and copies
up to k following tokens from the longest continuation found. Finished siblings keep their
text until the whole group is gone, so a straggler can copy from completed reasoning: the
tail case. This is the same rule as ``longtail.specsim`` ("both" source), so offline and
online numbers are directly comparable.

Under the gate the drafter returns no drafts, so vLLM schedules no verification tokens and
the bulk phase pays nothing.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

_GROUP_RE = re.compile(r"^g(\d+)-")


class _Seq:
    __slots__ = ("tokens", "index", "group", "active", "last_seen")

    def __init__(self, group: str | None, min_n: int, max_n: int):
        self.tokens: list[int] = []
        self.index: dict[int, dict[tuple, int]] = {n: {} for n in range(min_n, max_n + 1)}
        self.group = group
        self.active = True
        self.last_seen = 0

    def reset(self) -> None:
        self.tokens = []
        for d in self.index.values():
            d.clear()

    def extend(self, new: list[int]) -> None:
        old = len(self.tokens)
        toks = self.tokens
        toks.extend(new)
        # Index the n-gram ending just before position e once token e (its continuation) exists.
        for e in range(old, len(toks)):
            for n, d in self.index.items():
                if e >= n:
                    d[tuple(toks[e - n : e])] = e  # later occurrences overwrite earlier ones

    def lookup(self, key: tuple, k: int) -> list[int]:
        e = self.index[len(key)].get(key)
        return self.tokens[e : e + k] if e is not None else []


def _length(x) -> int:
    try:
        return len(x)
    except TypeError:
        return 0


class SiblingProposer:
    def __init__(self, vllm_config=None, *, k: int | None = None, max_model_len: int | None = None,
                 sources: str | None = None, gate: int | None = None,
                 min_n: int | None = None, max_n: int | None = None, evict_after: int = 1000):
        env = os.environ
        if vllm_config is not None:
            k = k or vllm_config.speculative_config.num_speculative_tokens
            max_model_len = max_model_len or vllm_config.model_config.max_model_len
        self.k = k or 8
        self.max_model_len = max_model_len or 1 << 30
        self.sources = sources or env.get("LONGTAIL_DRAFT_SOURCES", "both")
        if self.sources not in ("self", "sibling", "both"):
            raise ValueError(f"LONGTAIL_DRAFT_SOURCES must be self|sibling|both, got {self.sources!r}")
        self.gate = gate if gate is not None else int(env.get("LONGTAIL_DRAFT_GATE", "0"))
        self.min_n = min_n or int(env.get("LONGTAIL_DRAFT_MIN_N", "2"))
        self.max_n = max_n or int(env.get("LONGTAIL_DRAFT_MAX_N", "4"))
        self.evict_after = evict_after
        self.use_self = self.sources in ("self", "both")
        self.use_sib = self.sources in ("sibling", "both")

        self.seqs: dict[str, _Seq] = {}
        self.groups: dict[str, list[str]] = defaultdict(list)
        self.step = 0
        self.stats = defaultdict(int)
        self._logged_signature = False
        self._log(f"SiblingProposer: sources={self.sources} gate={self.gate} k={self.k} "
                  f"n={self.min_n}..{self.max_n}")

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[longtail.drafter] {msg}", file=sys.stderr, flush=True)

    def load_model(self, *args, **kwargs) -> None:
        pass

    # vLLM's call signature for custom proposers is not a stable API, so find the pieces
    # by shape: the InputBatch (has req_ids + token_ids_cpu), sampled ids (list), k (int).
    def propose(self, *args, **kwargs) -> list[list[int]]:
        batch = sampled = num_spec = None
        for a in list(args) + list(kwargs.values()):
            if batch is None and hasattr(a, "req_ids") and hasattr(a, "token_ids_cpu"):
                batch = a
            elif sampled is None and isinstance(a, list):
                sampled = a
            elif num_spec is None and isinstance(a, int) and not isinstance(a, bool):
                num_spec = a
        if not self._logged_signature:
            self._log("propose() args: " + ", ".join(type(a).__name__ for a in args)
                      + (f"; kwargs: {sorted(kwargs)}" if kwargs else ""))
            self._logged_signature = True
        if batch is None:
            raise TypeError("SiblingProposer.propose: no InputBatch-like argument (needs .req_ids and "
                            ".token_ids_cpu); vLLM's custom proposer interface may have changed")
        return self.propose_batch(num_spec, batch.req_ids, batch.token_ids_cpu,
                                  batch.num_tokens_no_spec, sampled)

    def propose_batch(self, num_spec, req_ids, token_ids, num_tokens, sampled) -> list[list[int]]:
        self.step += 1
        k = min(self.k, num_spec) if num_spec else self.k
        n_req = len(sampled) if sampled is not None else len(req_ids)
        rows: list[tuple[int, str, _Seq]] = []
        seen = set()

        # 1. Mirror every request's tokens (always, even when gated, so indices stay current).
        for i in range(n_req):
            rid = req_ids[i]
            if rid is None:
                continue
            seen.add(rid)
            seq = self.seqs.get(rid)
            if seq is None:
                m = _GROUP_RE.match(rid)
                group = m.group(1) if m else None
                seq = self.seqs[rid] = _Seq(group, self.min_n, self.max_n)
                if group is not None:
                    self.groups[group].append(rid)
            nt = int(num_tokens[i])
            if nt < len(seq.tokens):
                seq.reset()
            if nt > len(seq.tokens):
                seq.extend([int(t) for t in token_ids[i, len(seq.tokens) : nt]])
            seq.active = True
            seq.last_seen = self.step
            rows.append((i, rid, seq))

        if self.step % 256 == 0:
            self._evict(seen)

        drafts: list[list[int]] = [[] for _ in range(n_req)]
        decoding = [r for r in rows if sampled is None or _length(sampled[r[0]]) > 0]
        if self.gate and len(decoding) > self.gate:
            self.stats["gated_steps"] += 1
            return drafts
        self.stats["draft_steps"] += 1

        # 2. Draft for every decoding request.
        for i, rid, seq in decoding:
            kk = min(k, self.max_model_len - len(seq.tokens) - 1)
            if kk <= 0:
                continue
            draft, src = self._draft(rid, seq, kk)
            drafts[i] = draft
            if draft:
                self.stats[f"from_{src}"] += 1
                self.stats["drafted_tokens"] += len(draft)
            self.stats["requests"] += 1

        if self.step % 2000 == 0:
            s = self.stats
            self._log(f"step {self.step}: draft_steps={s['draft_steps']} gated={s['gated_steps']} "
                      f"from_self={s['from_self']} from_sibling={s['from_sibling']} "
                      f"drafted_tokens={s['drafted_tokens']} tracked={len(self.seqs)}")
        return drafts

    def _draft(self, rid: str, seq: _Seq, k: int) -> tuple[list[int], str]:
        toks = seq.tokens
        length = len(toks)
        siblings = self.groups.get(seq.group, ()) if (self.use_sib and seq.group is not None) else ()
        for n in range(self.max_n, self.min_n - 1, -1):
            if length < n:
                continue
            key = tuple(toks[length - n :])
            best, src = [], ""
            if self.use_self:
                c = seq.lookup(key, k)
                if len(c) > len(best):
                    best, src = c, "self"
            for sid in siblings:
                if sid != rid:
                    c = self.seqs[sid].lookup(key, k)
                    if len(c) > len(best):
                        best, src = c, "sibling"
            if best:
                return best, src
        return [], ""

    def _evict(self, seen: set[str]) -> None:
        for rid, seq in self.seqs.items():
            if rid not in seen:
                seq.active = False
        stale = self.step - self.evict_after
        for group, members in list(self.groups.items()):
            if all(not self.seqs[m].active and self.seqs[m].last_seen < stale for m in members):
                for m in members:
                    del self.seqs[m]
                del self.groups[group]
        for rid in [r for r, s in self.seqs.items() if s.group is None and not s.active and s.last_seen < stale]:
            del self.seqs[rid]
