import torch

from tools.train_plan_a import restore_unfrozen_blocks


def test_resume_restores_every_previously_unfrozen_block():
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    for block in blocks:
        block.requires_grad_(False)
    restore_unfrozen_blocks(blocks, 3)
    assert [block.weight.requires_grad for block in blocks] == [False, True, True, True]
    assert sum(p.numel() for p in blocks.parameters() if p.requires_grad) == 18


def test_resume_restores_full_backbone_and_rejects_bad_counts():
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    for block in blocks:
        block.requires_grad_(False)
    restore_unfrozen_blocks(blocks, 4)
    assert all(p.requires_grad for p in blocks.parameters())
    for count in (-1, 5):
        try:
            restore_unfrozen_blocks(blocks, count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid restored block count must fail")
