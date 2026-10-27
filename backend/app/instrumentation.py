"""Selective internal activation recording for one model forward pass."""
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
from torch import nn


class HeadAblation:
    """Temporarily zero one attention head before the output projection."""

    def __init__(self, model: nn.Module, layer: int, head: int) -> None:
        self.model = model
        self.layer = layer
        self.head = head
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        self.num_heads = int(model.config.num_attention_heads)
        self.handle = None
        self.path = None

    def start(self) -> None:
        if self.head < 0 or self.head >= self.num_heads:
            raise ValueError(f"head must be between 0 and {self.num_heads - 1}")
        suffix = f"layers.{self.layer}.self_attn.o_proj"
        for path, module in self.model.named_modules():
            if path.endswith(suffix):
                self.path = path

                def mask_head(_module: nn.Module, args: tuple):
                    hidden = args[0].clone()
                    start = self.head * self.head_dim
                    hidden[..., start:start + self.head_dim] = 0
                    return (hidden, *args[1:])

                self.handle = module.register_forward_pre_hook(mask_head)
                return
        raise ValueError(f"attention output projection not found for layer {self.layer}")

    def stop(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TraceRecorder:
    """Attach temporary hooks to selected transformer components."""

    COMPONENT_ALIASES = {
        "hidden": "hidden",
        "hidden_states": "hidden",
        "attention": "attention",
        "attn": "attention",
        "mlp": "mlp",
        "intermediate": "intermediate",
        "logits": "logits",
    }

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[Iterable[int]] = None,
        components: Optional[Iterable[str]] = None,
    ) -> None:
        self.model = model
        self.layer_count = self._find_layer_count()
        requested_layers = list(layer_indices) if layer_indices is not None else [0, self.layer_count - 1]
        self.layer_indices: List[int] = sorted({layer for layer in requested_layers if 0 <= layer < self.layer_count})
        requested_components = components or ["hidden", "attention", "mlp", "intermediate"]
        self.components: Set[str] = {
            self.COMPONENT_ALIASES.get(component.lower(), component.lower())
            for component in requested_components
            if component.lower() != "logits"
        }
        self.trace_id = str(uuid.uuid4())
        self.records: List[Dict[str, Any]] = []
        self.architecture: List[Dict[str, str]] = []
        self._handles: List[Any] = []
        self._module_paths: Dict[str, str] = {}
        self._inspect_architecture()

    def _find_layer_count(self) -> int:
        layers = set()
        for name, _ in self.model.named_modules():
            match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", name)
            if match:
                layers.add(int(match.group(1)))
        return max(layers) + 1 if layers else 0

    def _inspect_architecture(self) -> None:
        for name, module in self.model.named_modules():
            match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.(.*))?$", name)
            if not match:
                continue
            layer = int(match.group(1))
            suffix = match.group(2) or "block"
            kind = "other"
            lowered = suffix.lower()
            if "self_attn" in lowered or "attention" in lowered or lowered == "attn":
                kind = "attention"
            elif lowered.endswith("mlp") or ".mlp" in lowered:
                kind = "mlp"
            elif "norm" in lowered:
                kind = "normalization"
            self.architecture.append({"path": name, "layer": str(layer), "component": kind, "type": module.__class__.__name__})

    def _component_for_path(self, path: str) -> Optional[str]:
        lowered = path.lower()
        if path.endswith(".self_attn") or path.endswith(".attention") or path.endswith(".attn"):
            return "attention"
        if path.endswith(".mlp"):
            return "mlp"
        if re.search(r"(?:^|\.)(?:layers|h)\.\d+$", path):
            return "hidden"
        if path.endswith(".intermediate"):
            return "intermediate"
        return None

    def _layer_for_path(self, path: str) -> Optional[int]:
        match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", path)
        return int(match.group(1)) if match else None

    @staticmethod
    def _first_tensor(value: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = TraceRecorder._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _attention_tensor(value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, (tuple, list)):
            for item in value:
                if torch.is_tensor(item) and item.dim() == 4:
                    return item
        return None

    def _record_hook(self, path: str, module: nn.Module, _inputs: Any, output: Any) -> None:
        layer = self._layer_for_path(path)
        component = self._component_for_path(path)
        if layer is None or component is None or layer not in self.layer_indices or component not in self.components:
            return
        tensor = self._attention_tensor(output) if component == "attention" else self._first_tensor(output)
        # Some modern attention implementations return only the projected output;
        # retain that tensor even when attention weights are not exposed.
        if tensor is None and component == "attention":
            tensor = self._first_tensor(output)
        if tensor is None:
            return
        detached = tensor.detach().float().cpu()
        record: Dict[str, Any] = {
            "path": path,
            "layer": layer,
            "component": component,
            "kind": "weights" if component == "attention" and detached.dim() == 4 else "output",
            "shape": list(detached.shape),
            "dtype": str(tensor.dtype),
            "mean": round(float(detached.mean()), 6),
            "std": round(float(detached.std()), 6),
            "min": round(float(detached.min()), 6),
            "max": round(float(detached.max()), 6),
            "abs_max": round(float(detached.abs().max()), 6),
        }
        # Keep visualizable matrices bounded; summaries remain available for larger tensors.
        if component == "attention" and detached.dim() == 4:
            record["values"] = detached[0, 0].tolist()
        elif detached.dim() == 3 and detached.shape[0] > 0:
            record["token_means"] = detached[0].abs().mean(dim=-1).tolist()
        self.records.append(record)
        print(f"[trace {self.trace_id[:8]}] captured {path} shape={record['shape']} component={component}")

    def start(self) -> None:
        for path, module in self.model.named_modules():
            layer = self._layer_for_path(path)
            component = self._component_for_path(path)
            if layer is None or component is None or layer not in self.layer_indices or component not in self.components:
                continue
            self._module_paths[f"{layer}:{component}"] = path
            self._handles.append(module.register_forward_hook(lambda m, i, o, p=path: self._record_hook(p, m, i, o)))

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def build_trace(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        logits_cpu = logits.detach().float().cpu()
        self.records.append({
            "path": "lm_head",
            "layer": -1,
            "component": "logits",
            "shape": list(logits_cpu.shape),
            "dtype": str(logits.dtype),
            "mean": round(float(logits_cpu.mean()), 6),
            "std": round(float(logits_cpu.std()), 6),
            "min": round(float(logits_cpu.min()), 6),
            "max": round(float(logits_cpu.max()), 6),
        })
        captured = [f"{item['component']}:{item['path']} {item['shape']}" for item in self.records]
        print(f"[trace {self.trace_id[:8]}] complete tensors={len(captured)}")
        return {
            "trace_id": self.trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_shape": list(input_ids.shape),
            "layer_count": self.layer_count,
            "selected_layers": self.layer_indices,
            "selected_components": sorted(self.components | {"logits"}),
            "architecture": self.architecture,
            "captures": self.records,
            "capture_log": captured,
        }

    def add_forward_outputs(self, hidden_states: Optional[tuple], attentions: Optional[tuple]) -> None:
        """Add model-returned tensors, preserving layer/head dimensions."""
        if hidden_states and "hidden" in self.components:
            for layer in self.layer_indices:
                if layer + 1 >= len(hidden_states):
                    continue
                tensor = hidden_states[layer + 1].detach().float().cpu()
                self.records.append({
                    "path": f"hidden_states.{layer + 1}", "layer": layer, "component": "hidden",
                    "kind": "state", "shape": list(tensor.shape), "dtype": str(hidden_states[layer + 1].dtype),
                    "mean": round(float(tensor.mean()), 6), "std": round(float(tensor.std()), 6),
                    "min": round(float(tensor.min()), 6), "max": round(float(tensor.max()), 6),
                    "abs_max": round(float(tensor.abs().max()), 6),
                    "token_means": tensor[0].abs().mean(dim=-1).tolist(),
                })
        if attentions and "attention" in self.components:
            for layer in self.layer_indices:
                if layer >= len(attentions) or attentions[layer] is None:
                    continue
                tensor = attentions[layer].detach().float().cpu()
                entropy = -(tensor.clamp_min(1e-9) * tensor.clamp_min(1e-9).log()).sum(dim=-1)
                strongest = tensor.mean(dim=1).mean(dim=1).argmax(dim=-1).tolist()
                self.records.append({
                    "path": f"attentions.{layer}", "layer": layer, "component": "attention_weights",
                    "kind": "weights", "shape": list(tensor.shape), "dtype": str(attentions[layer].dtype),
                    "mean": round(float(tensor.mean()), 6), "std": round(float(tensor.std()), 6),
                    "min": round(float(tensor.min()), 6), "max": round(float(tensor.max()), 6),
                    "abs_max": round(float(tensor.abs().max()), 6),
                    "values": tensor[0].tolist(),
                    "mean_entropy": round(float(entropy.mean()), 6),
                    "strongest_key_by_query": strongest,
                })

    def attention_record(self, layer: int) -> Optional[Dict[str, Any]]:
        return next((record for record in self.records if record["layer"] == layer and record["component"] == "attention_weights"), None)
