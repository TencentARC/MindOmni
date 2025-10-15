import torch
import torch.nn.functional as F
from peft import PeftModel


def is_peft_model(model):
    if hasattr(model, "module"):
        if isinstance(model.module, PeftModel):
            return True
        else:
            return False
    else:
        if isinstance(model, PeftModel):
            return True
        else:
            return False

def sample_x0(x1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [torch.randn_like(img_start) for img_start in x1]
    else:
        x0 = torch.randn_like(x1)

    return x0

def sample_timestep(x1):
    u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
    t = 1 / (1 + torch.exp(-u))
    t = t.to(x1[0])
    return t


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kl_div(input, target, reduction='batchmean'):
    log_p = torch.log(target + 1e-8)
    if reduction == "batchmean":
        result = torch.sum(target * (log_p - input)) / target.shape[0]  # 手动 forward KL
    elif reduction == "mean":
        result = torch.sum(target * (log_p - input), dim=-1).mean()  # 手动 forward KL
    return result


def training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None, use_dist=False):
    """Loss for training torche score model
    Args:
    - model: backbone model; could be score, noise, or velocity
    - x1: datapoint
    - model_kwargs: additional arguments for torch model
    """
    if model_kwargs == None:
        model_kwargs = {}

    B = len(x1)

    x0 = sample_x0(x1)
    t = sample_timestep(x1)

    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    model_output, all_hidden_states = model(xt, t, **model_kwargs, use_dist=use_dist)

    terms = {}

    if use_dist:
        with torch.no_grad():
            _ = model_kwargs.pop("llm_input_embeds")
            _ = model_kwargs.pop("llm_attention_mask")
            _ = model_kwargs.pop("llm_position_ids")

            if is_peft_model(model):
                if hasattr(model, "module"):     # ddp mode
                    with model.module.disable_adapter():
                        model_output_tea, all_hidden_states_tea = model(xt, t, **model_kwargs, use_dist=use_dist)
                else:
                    with model.disable_adapter():
                        model_output_tea, all_hidden_states_tea = model(xt, t, **model_kwargs, use_dist=use_dist)
            else:
                model_output_tea, all_hidden_states_tea = model(xt, t, **model_kwargs, use_dist=use_dist)
        if isinstance(xt, (list, tuple)):    # when use raw image resolution ratio
            temperature0 = 3
            dist_loss = []
            for i in range(len(all_hidden_states)):
                batch_dist_loss = F.kl_div(F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1).log(), F.softmax(normalize(all_hidden_states_tea[i])/temperature0, dim=-1), reduction='batchmean')
                # ## reverse kl
                # batch_dist_loss = kl_div(F.softmax(normalize(all_hidden_states_tea[i])/temperature0, dim=-1).log(), F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1), reduction='batchmean')
                # batch_dist_loss = F.kl_div(F.softmax(normalize(all_hidden_states_tea[i])/temperature0, dim=-1).log(), F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1), reduction='batchmean')

                dist_loss.append(batch_dist_loss)
            terms['dist_loss'] = torch.stack(dist_loss).mean()
        else:
            temperature0 = 3
            dist_loss = []
            for i in range(len(all_hidden_states)):
                batch_dist_loss = F.kl_div(F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1).log(), F.softmax(normalize(all_hidden_states_tea[i])/temperature0, dim=-1), reduction='batchmean')
                # ## reverse kl
                # batch_dist_loss = kl_div(F.softmax(normalize(all_hidden_states_tea[i])/temperature0, dim=-1).log(), F.softmax(normalize(all_hidden_states[i])/temperature0, dim=-1), reduction='batchmean')

                dist_loss.append(batch_dist_loss)
            terms['dist_loss'] = torch.stack(dist_loss).mean()

    if isinstance(x1, (list, tuple)):
        assert len(model_output) == len(ut) == len(x1)
        if patch_weight is not None:
            terms["loss"] = torch.stack(
            [((ut[i] - model_output[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
            dim=0,
            ).mean()
        else:
            terms["loss"] = torch.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
            ).mean()

            # ## add diffusion focal loss (under developping)
            # loss = []
            # for i in range(B):
            #     # focal_weight = focal_patch_weight(model_output[i], ut[i])
            #     # loss.append(((ut[i] - model_output[i]) ** 2 * focal_weight).mean())

            #     kl_loss = kl_diffusion_loss(model_output[i], ut[i], temperature0=3.0)
            #     loss.append(((ut[i] - model_output[i]) ** 2).mean() * 0.5 + kl_loss * 0.5)
            # terms["loss"] = torch.stack(loss, dim=0).mean()
    else:
        if patch_weight is not None:
            loss = (model_output - ut) ** 2
            loss = loss * patch_weight
            terms["loss"] = mean_flat(loss)
        else:
            terms["loss"] = mean_flat(((model_output - ut) ** 2))

    return terms


def focal_patch_weight(pred, target, tau=1.0, gamma=2.0):
    diff = pred - target
    abs_diff = torch.abs(diff)
    weights = (abs_diff / tau).clamp(min=1e-6) ** gamma

    return weights


def kl_diffusion_loss(pred, target, temperature0=3.0):
    # pred = pred.permute(0, 2, 3, 1)
    # target = target.permute(0, 2, 3, 1)

    b, c, h, w = pred.shape
    pred = pred.view(b, c, h * w)
    target = target.view(b, c, h * w)
    kl_loss = F.kl_div(F.softmax(normalize(pred)/temperature0, dim=-1).log(), F.softmax(normalize(target)/temperature0, dim=-1), reduction='batchmean')

    return kl_loss


def mean_flat(x):
    """
    Take torche mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))
