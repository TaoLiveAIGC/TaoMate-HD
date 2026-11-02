# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Opt-in, request-local condition reuse for LOW TP1/U1 inference.

The caller supplies identical-condition keys, including each document's actual
timestep, and clears the bank when the prompt/request changes. Media attention
still receives the original full Q/K/V row order. A hit's Q is unused because
the streaming hook skips its condition attention; only raw BF16 K/V and the
block output are retained. Shorter GEMMs may change floating-point rounding.
"""

from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from typing import Any, Hashable, Iterator, Sequence


@dataclass(frozen=True)
class ConditionCacheDocument:
    key: Hashable
    rows: Any
    document_index: int = 0


@dataclass
class _Entry:
    key: Any
    value: Any
    hidden: Any


class ConditionCacheBank:
    def __init__(self) -> None:
        self.entries: dict[tuple[Hashable, str], _Entry] = {}
        self.layer_hits = 0
        self.layer_misses = 0
        self.reused_rows = 0

    def clear(self) -> None:
        self.entries.clear()

    def stats(self) -> dict[str, int]:
        return {
            "layer_hits": self.layer_hits,
            "layer_misses": self.layer_misses,
            "reused_rows": self.reused_rows,
            "entries": len(self.entries),
            "cached_bytes": sum(
                (
                    tensor.numel() * tensor.element_size()
                    for entry in self.entries.values()
                    for tensor in (entry.key, entry.value, entry.hidden)
                )
            ),
        }


class _LayerPlan:
    def __init__(self, scope: _Scope, name: str, value: Any) -> None:
        import torch

        (self.scope, self.name) = (scope, name)
        self.total = scope.seq_len
        self.hits = []
        self.misses = []
        self.pending: dict[Hashable, tuple[Any, Any]] = {}
        self.attention_output_compact = False
        for document in scope.documents:
            entry = scope.bank.entries.get((document.key, name))
            if entry is None:
                self.misses.append(document)
                scope.bank.layer_misses += 1
            else:
                self.hits.append((document, entry))
                scope.bank.layer_hits += 1
                scope.bank.reused_rows += int(document.rows.numel())
        self.active_rows = None
        if self.hits:
            hit_documents = tuple((document.document_index for (document, _) in self.hits))
            self.active_rows = scope.active_rows.get(hit_documents)
            if self.active_rows is None:
                keep = torch.ones(self.total, dtype=torch.bool, device=value.device)
                for document, _ in self.hits:
                    keep.index_fill_(0, document.rows, False)
                self.active_rows = torch.nonzero(keep, as_tuple=False).view(-1)
                scope.active_rows[hit_documents] = self.active_rows

    def select(self, value: Any) -> Any:
        return value if self.active_rows is None else value.index_select(0, self.active_rows)

    def rope(self, cache: tuple[Any, Any]) -> tuple[Any, Any]:
        if self.scope.compact:
            return cache
        (cos_sin, positions) = cache
        return (cos_sin, self.select(positions))

    def select_attention_output(self, value: Any) -> Any:
        return value if self.attention_output_compact else self.select(value)

    def full_qkv(self, q: Any, k: Any, v: Any) -> tuple[Any, Any, Any]:
        if self.active_rows is not None:
            if self.scope.fused_qkv_restore:
                from .condition_cache_kernels import fused_restore_condition_qkv

                (q, k, v) = fused_restore_condition_qkv(
                    q,
                    k,
                    v,
                    active_rows=self.active_rows,
                    total_rows=self.total,
                    cached_parts=tuple(
                        ((document.rows, entry.key, entry.value) for (document, entry) in self.hits)
                    ),
                )
            else:
                full = q.new_empty((self.total, 3, *q.shape[1:]))
                (full_q, full_k, full_v) = full.unbind(dim=1)
                for target, source in ((full_q, q), (full_k, k), (full_v, v)):
                    target.index_copy_(0, self.active_rows, source)
                for document, entry in self.hits:
                    full_q.index_fill_(0, document.rows, 0)
                    full_k.index_copy_(0, document.rows, entry.key)
                    full_v.index_copy_(0, document.rows, entry.value)
                (q, k, v) = (full_q, full_k, full_v)
        for document in self.misses:
            self.pending[document.key] = (
                k.detach().index_select(0, document.rows),
                v.detach().index_select(0, document.rows),
            )
        return (q, k, v)

    def finish(self, hidden: Any) -> Any:
        if self.active_rows is not None and (not self.scope.compact):
            full = hidden.new_empty((self.total, *hidden.shape[1:]))
            full.index_copy_(0, self.active_rows, hidden)
            for document, entry in self.hits:
                full.index_copy_(0, document.rows, entry.hidden)
            hidden = full
        for document in self.misses:
            (key, value) = self.pending[document.key]
            rows = self.scope.compact_document_rows.get(document.document_index, document.rows)
            self.scope.bank.entries[document.key, self.name] = _Entry(
                key, value, hidden.detach().index_select(0, rows)
            )
        self.scope.layer = None
        return hidden


class _Scope:
    def __init__(
        self, bank: ConditionCacheBank, documents: Sequence[ConditionCacheDocument], seq_len: int
    ) -> None:
        (self.bank, self.documents, self.seq_len) = (bank, tuple(documents), seq_len)
        self.layer: _LayerPlan | None = None
        self.active_rows: dict[tuple[int, ...], Any] = {}
        self.compact = os.environ.get("H3_PROGRESSIVE_LOW_CONDITION_COMPACT") == "1"
        self.fused_qkv_restore = os.environ.get("H3_PROGRESSIVE_LOW_FUSED_QKV_RESTORE") == "1"
        self.compact_attention_output = (
            os.environ.get("H3_PROGRESSIVE_LOW_COMPACT_ATTN_OUTPUT") == "1"
        )
        self.attention_output_rows: dict[tuple[Any, ...], Any] = {}
        self.global_to_compact: Any | None = None
        self.compact_document_rows: dict[int, Any] = {}
        self.compact_active_rows: Any | None = None
        self.compact_hit_documents: tuple[ConditionCacheDocument, ...] = ()


_ACTIVE: ContextVar[_Scope | None] = ContextVar("h3_condition_cache", default=None)


class _CompactAttentionOutput:
    """Assemble attention rows directly in the cross-layer compact order."""

    def __init__(self, scope: _Scope, query: Any) -> None:
        import torch

        self.scope = scope
        active = scope.compact_active_rows
        self.output = query.new_zeros((int(active.numel()), *query.shape[1:]))
        if scope.global_to_compact is None:
            scope.global_to_compact = torch.empty(
                scope.seq_len, dtype=torch.long, device=query.device
            )
            scope.global_to_compact.index_copy_(
                0, active, torch.arange(active.numel(), device=query.device)
            )

    def write_rows(self, document: int, kind: str, start: int, rows: Any, value: Any) -> None:
        key = (document, kind, start)
        destination = self.scope.attention_output_rows.get(key)
        if destination is None:
            destination = self.scope.global_to_compact.index_select(0, rows + start)
            self.scope.attention_output_rows[key] = destination
        self.output.index_copy_(0, destination, value)

    def write_document(self, start: int, stop: int, value: Any) -> None:
        import torch

        key = ("document", start, stop)
        selected = self.scope.attention_output_rows.get(key)
        if selected is None:
            active = self.scope.compact_active_rows
            destination = torch.nonzero((active >= start) & (active < stop), as_tuple=False).view(
                -1
            )
            source = active.index_select(0, destination) - start
            selected = (destination, source)
            self.scope.attention_output_rows[key] = selected
        (destination, source) = selected
        self.output.index_copy_(0, destination, value.index_select(0, source))


def compact_condition_attention_output(
    layer_name: str, query: Any
) -> _CompactAttentionOutput | None:
    scope = _ACTIVE.get()
    plan = None if scope is None else scope.layer
    if (
        plan is None
        or not scope.compact_attention_output
        or (not scope.compact)
        or (scope.compact_active_rows is None)
        or (plan.name != layer_name.removesuffix(".attn"))
    ):
        return None
    plan.attention_output_compact = True
    return _CompactAttentionOutput(scope, query)


@contextmanager
def condition_cache_scope(
    bank: ConditionCacheBank, *, documents: Sequence[ConditionCacheDocument], seq_len: int
) -> Iterator[None]:
    token = _ACTIVE.set(_Scope(bank, documents, int(seq_len)))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def require_condition_cache_inference(model: Any, seq_len: int) -> None:
    scope = _ACTIVE.get()
    if scope is None:
        return
    import torch

    context = model.parallel_context
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("condition cache requires eval/no-grad inference")
    if context.tp_world_size != 1 or context.ulysses_world_size != 1:
        raise RuntimeError("condition cache currently requires LOW TP1/U1")
    if scope.seq_len != seq_len:
        raise ValueError("condition cache sequence length differs")


def compact_condition_cache_inputs(
    hidden: Any, combined_indices: Any, rope_cache: tuple[Any, Any], *, first_layer_name: str
) -> tuple[Any, Any, tuple[Any, Any]]:
    """Select fixed live rows once; misses retain their condition rows."""
    scope = _ACTIVE.get()
    if scope is None or not scope.compact:
        return (hidden, combined_indices, rope_cache)
    import torch

    hits = tuple(
        (
            document
            for document in scope.documents
            if (document.key, first_layer_name) in scope.bank.entries
        )
    )
    if not hits:
        return (hidden, combined_indices, rope_cache)
    keep = torch.ones(scope.seq_len, dtype=torch.bool, device=hidden.device)
    for document in hits:
        keep.index_fill_(0, document.rows, False)
    active = torch.nonzero(keep, as_tuple=False).view(-1)
    scope.active_rows[tuple((document.document_index for document in hits))] = active
    scope.compact_active_rows = active
    scope.compact_hit_documents = hits
    if len(hits) != len(scope.documents):
        global_to_compact = torch.empty(scope.seq_len, dtype=torch.long, device=hidden.device)
        global_to_compact.index_copy_(0, active, torch.arange(active.numel(), device=hidden.device))
        if scope.compact_attention_output:
            scope.global_to_compact = global_to_compact
        hit_indices = {document.document_index for document in hits}
        for document in scope.documents:
            if document.document_index not in hit_indices:
                scope.compact_document_rows[document.document_index] = (
                    global_to_compact.index_select(0, document.rows)
                )
    (cos_sin, positions) = rope_cache
    return (
        hidden.index_select(0, active),
        combined_indices.index_select(0, active),
        (cos_sin, positions.index_select(0, active)),
    )


def restore_condition_cache_output(hidden: Any, *, last_layer_name: str) -> Any:
    """Restore only the final block's condition state for the output heads."""
    scope = _ACTIVE.get()
    if scope is None or scope.compact_active_rows is None:
        return hidden
    full = hidden.new_empty((scope.seq_len, *hidden.shape[1:]))
    full.index_copy_(0, scope.compact_active_rows, hidden)
    for document in scope.compact_hit_documents:
        entry = scope.bank.entries[document.key, last_layer_name]
        full.index_copy_(0, document.rows, entry.hidden)
    return full


def condition_layer_plan(name: str, value: Any) -> _LayerPlan | None:
    scope = _ACTIVE.get()
    if scope is None:
        return None
    scope.layer = _LayerPlan(scope, name, value)
    return scope.layer


def condition_attention_cached(layer_name: str, document_index: int) -> bool:
    scope = _ACTIVE.get()
    plan = None if scope is None else scope.layer
    return bool(
        plan is not None
        and plan.name == layer_name.removesuffix(".attn")
        and any((document.document_index == document_index for (document, _) in plan.hits))
    )
