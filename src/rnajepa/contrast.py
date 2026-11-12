from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ResidueInfoNCE(nn.Module):
    def __init__(self, d_model=1280, hidden=256, d_output=128, temperature=0.1,
                 max_negatives=32, queue_size=256):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                        nn.Linear(hidden, d_output))
        self.temperature = temperature
        self.max_negatives = max_negatives
        self.queue_size = queue_size
        self.queue_z = None
        self.queue_hashes = []
        self.queue_legal = []

    def pooled(self, hidden, lengths):
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
        return (hidden * mask[:, :, None]).sum(dim=1) / lengths[:, None].to(hidden.dtype)

    def project(self, hidden, lengths):
        return self.projection(self.pooled(hidden, lengths))

    def candidates(self, z, hashes, legal):
        tensors, names = [], []
        if self.queue_z is not None:
            for index, (name, allowed) in enumerate(zip(self.queue_hashes, self.queue_legal)):
                if allowed:
                    tensors.append(self.queue_z[index])
                    names.append(name)
        for index, (name, allowed) in enumerate(zip(hashes, legal)):
            if allowed:
                tensors.append(z[index])
                names.append(name)
        return (torch.stack(tensors) if tensors else None), names

    def direction(self, anchors, positives, anchor_hashes, positive_hashes, legal):
        candidates, names = self.candidates(positives, positive_hashes, legal)
        losses, negative_scores, used_names = [], [], []
        for index, (anchor, positive, sequence) in enumerate(zip(anchors, positives, anchor_hashes)):
            keep = [j for j, name in enumerate(names) if name != sequence]
            if candidates is None or not keep:
                raise ValueError("InfoNCE requires at least one legal negative")
            selected = keep[:self.max_negatives]
            negatives = candidates[selected]
            negative = F.normalize(negatives, dim=-1)
            pos = torch.dot(anchors[index], F.normalize(positive, dim=0)) / self.temperature
            neg = negative @ F.normalize(anchor, dim=0) / self.temperature
            losses.append(-pos + torch.logsumexp(torch.cat([pos[None], neg]), dim=0))
            negative_scores.append(neg.detach().mean())
            used_names.extend(names[j] for j in selected)
        return (torch.stack(losses).mean(), torch.stack(negative_scores).mean(), used_names)

    def forward(self, h1, h2, lengths, sequence_hashes, legal_by_hash, log_details=False):
        raw1, raw2 = self.project(h1, lengths), self.project(h2, lengths)
        if self.queue_z is not None:
            self.queue_z = self.queue_z.to(raw1.device)
        z1, z2 = F.normalize(raw1, dim=-1), F.normalize(raw2, dim=-1)
        legal = [bool(legal_by_hash.get(sequence, False)) for sequence in sequence_hashes]
        loss1, neg1, names1 = self.direction(z1, z2, sequence_hashes, sequence_hashes, legal)
        loss2, neg2, names2 = self.direction(z2, z1, sequence_hashes, sequence_hashes, legal)
        positive = (z1 * z2).sum(dim=-1).mean().detach()
        diagnostics = {
            "positive_similarity": float(positive),
            "negative_similarity": float((neg1 + neg2).detach() / 2),
            "representation_norm": float(torch.cat([raw1, raw2]).norm(dim=-1).mean().detach()),
            "channel_variance": float(torch.cat([raw1, raw2]).var(dim=0, unbiased=False).mean().detach()),
            "negative_count": len(names1) + len(names2),
        }
        if log_details:
            singular = torch.linalg.svdvals(torch.cat([raw1, raw2], dim=0))
            diagnostics["effective_rank"] = float((singular.square() / singular.square().sum()).sum().reciprocal())
            diagnostics["negative_sequence_sha256"] = names1 + names2
        current_z = torch.cat([z1, z2], dim=0).detach()
        if self.queue_z is None:
            self.queue_z = current_z
        else:
            self.queue_z = torch.cat([self.queue_z, current_z], dim=0)[-self.queue_size:]
        self.queue_hashes = (self.queue_hashes + list(sequence_hashes) * 2)[-self.queue_size:]
        self.queue_legal = (self.queue_legal + legal * 2)[-self.queue_size:]
        return (loss1 + loss2) / 2, diagnostics

    def export_state(self):
        return {"model": self.state_dict(), "queue_z": self.queue_z,
                "queue_hashes": self.queue_hashes, "queue_legal": self.queue_legal}

    def restore_state(self, state):
        self.load_state_dict(state["model"])
        self.queue_z = state["queue_z"]
        self.queue_hashes = state["queue_hashes"]
        self.queue_legal = state["queue_legal"]
