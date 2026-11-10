import torch


def loss_gradient_cosine(first_loss, second_loss, parameters):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("at least one trainable parameter is required")
    first_gradients = torch.autograd.grad(first_loss, parameters, retain_graph=True,
                                         allow_unused=True)
    second_gradients = torch.autograd.grad(second_loss, parameters, retain_graph=True,
                                          allow_unused=True)
    device = first_loss.device
    dot = torch.zeros((), dtype=torch.float64, device=device)
    first_norm = torch.zeros_like(dot)
    second_norm = torch.zeros_like(dot)
    first_count = second_count = shared_count = 0
    for parameter, first, second in zip(parameters, first_gradients, second_gradients):
        if first is not None:
            first_count += parameter.numel()
            first_norm += first.detach().double().square().sum()
        if second is not None:
            second_count += parameter.numel()
            second_norm += second.detach().double().square().sum()
        if first is not None and second is not None:
            shared_count += parameter.numel()
            dot += (first.detach().double() * second.detach().double()).sum()
    denominator = first_norm.sqrt() * second_norm.sqrt()
    cosine = dot / denominator if denominator > 0 else torch.zeros_like(dot)
    return {"cosine": float(cosine), "nll_gradient_norm": float(first_norm.sqrt()),
            "jepa_gradient_norm": float(second_norm.sqrt()),
            "nll_gradient_elements": first_count, "jepa_gradient_elements": second_count,
            "shared_gradient_elements": shared_count}
