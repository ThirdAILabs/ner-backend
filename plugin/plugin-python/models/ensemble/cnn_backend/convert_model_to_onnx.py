import json
import torch
import os
import torch.nn as nn
from .backend import CNNNERModelSentenceTokenized


class EmissionModel(nn.Module):

    def __init__(self, base: CNNNERModelSentenceTokenized):
        super().__init__()
        self.base = base

    def forward(self, input_ids: torch.Tensor):

        embeds = self.base.embedding(input_ids)  # (B, L, E)
        x = embeds.transpose(1, 2)  # (B, E, L)
        if self.base.proj is not None:
            x = self.base.proj(x)  # (B, C, L)

        out1, out2 = x, x
        for b in range(self.base.num_blocks):
            i1, i2 = 2 * b, 2 * b + 1

            y1 = self.base.branch1_convs[i1](out1)
            y1 = self.base.batch_norms_1[i1](y1)
            y1 = torch.nn.functional.leaky_relu(y1)
            y1 = self.base.branch1_convs[i2](y1)
            y1 = self.base.batch_norms_1[i2](y1)
            y1 = torch.nn.functional.leaky_relu(y1)
            out1 = x + y1

            z1 = self.base.branch2_convs[i1](out2)
            z1 = self.base.batch_norms_2[i1](z1)
            z1 = torch.nn.functional.leaky_relu(z1)
            z1 = self.base.branch2_convs[i2](z1)
            z1 = self.base.batch_norms_2[i2](z1)
            z1 = torch.nn.functional.leaky_relu(z1)
            out2 = x + z1

        combined = out1 + out2  # (B, C, L)
        combined = combined.transpose(1, 2)  # (B, L, C)
        emissions = self.base.hidden2tag(combined)  # (B, L, num_tags)
        return emissions


def export_to_onnx(
    model: CNNNERModelSentenceTokenized, onnx_out: str, max_seq_len: int = 128
):
    device = torch.device("cpu")
    model.eval()

    wrapper = EmissionModel(model).to(device)

    dummy = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        os.path.join(onnx_out, "cnn_emissions.onnx"),
        input_names=["input_ids"],
        output_names=["emissions"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "emissions": {0: "batch", 1: "seq"},
        },
        opset_version=13,
        do_constant_folding=True,
    )

    transitions = model.crf.transitions.detach().cpu().numpy().tolist()

    # some versions expose start_transitions & end_transitions
    start_probs = (
        model.crf.start_transitions.detach().cpu().numpy().tolist()
        if hasattr(model.crf, "start_transitions")
        else []
    )
    end_probs = (
        model.crf.end_transitions.detach().cpu().numpy().tolist()
        if hasattr(model.crf, "end_transitions")
        else []
    )

    with open(
        os.path.join(onnx_out, "crf_transitions.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "Transitions": transitions,
                "StartProbs": start_probs,
                "EndProbs": end_probs,
            },
            f,
            indent=2,
        )
