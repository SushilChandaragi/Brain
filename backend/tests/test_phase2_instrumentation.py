import torch
from torch import nn
from types import SimpleNamespace

from app.instrumentation import HeadAblation, TraceRecorder


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return self.o_proj(hidden)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return self.mlp(self.self_attn(hidden))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
        self.layers = nn.ModuleList([FakeLayer()])

    def forward(self, hidden):
        return self.layers[0](hidden)


def test_head_ablation_restores_module_behavior():
    model = FakeModel()
    inputs = torch.ones(1, 2, 8)
    baseline = model(inputs)
    intervention = HeadAblation(model, layer=0, head=1)
    intervention.start()
    ablated = model(inputs)
    intervention.stop()
    restored = model(inputs)

    assert not torch.equal(baseline, ablated)
    assert torch.equal(baseline, restored)
    assert torch.equal(model.layers[0].self_attn.o_proj.weight, model.layers[0].self_attn.o_proj.weight.detach())


def test_trace_preserves_attention_head_dimensions():
    model = FakeModel()
    recorder = TraceRecorder(model, layer_indices=[0], components=["attention"])
    attentions = (torch.rand(1, 2, 3, 3),)
    recorder.add_forward_outputs(None, attentions)

    capture = recorder.attention_record(0)
    assert capture is not None
    assert capture["shape"] == [1, 2, 3, 3]
    assert len(capture["values"]) == 2
