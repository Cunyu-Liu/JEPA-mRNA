import torch

from rnajepa.contrast import ResidueInfoNCE


def main():
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        torch.manual_seed(7)
        model = ResidueInfoNCE(d_model=8, hidden=12, d_output=6, max_negatives=4).to(device)
        lengths = torch.tensor([4, 3, 5, 2], device=device)
        h1 = torch.randn(4, 5, 8, device=device)
        h2 = torch.randn(4, 5, 8, device=device)
        hashes = ["a", "b", "c", "forbidden"]
        legal = {"a": True, "b": True, "c": True, "forbidden": False}
        loss, diagnostics = model(h1, h2, lengths, hashes, legal, log_details=True)
        assert torch.isfinite(loss) and diagnostics["negative_count"] > 0
        assert "forbidden" not in diagnostics["negative_sequence_sha256"]
        anchors = torch.randn(3, 6, device=device)
        _, _, negative_names = model.direction(anchors, anchors, ["a", "a", "a"],
                                                ["a", "b", "c"], [True, True, True])
        assert set(negative_names) == {"b", "c"} and "a" not in negative_names
        loss.backward()
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())
        restored = ResidueInfoNCE(d_model=8, hidden=12, d_output=6, max_negatives=4).to(device)
        restored.restore_state(model.export_state())
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected, _ = model(h1, h2, lengths, hashes, legal)
            actual, _ = restored(h1, h2, lengths, hashes, legal)
        torch.testing.assert_close(actual, expected)
        print("CL2_INFO_NCE_GRAD_NEGATIVE_RESUME_PASS", device, float(loss))


if __name__ == "__main__":
    main()
