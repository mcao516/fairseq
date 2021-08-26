# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, probs, epsilon, ignore_index=None, reduce=True):
    """
    Args:
        lprobs (Tensor): [batch_size * max_tgt_len, vocab_size]
        target (Tensor): [batch_size * max_tgt_len]
        probs (Tensor): [batch_size * max_tgt_len, vocab_size]
        p_mle (Tensor): [batch_size * max_tgt_len]
        
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1) # [batch_size * max_tgt_len, 1]
    nll_loss = -lprobs.gather(dim=-1, index=target) # [batch_size * max_tgt_len, 1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True) # [batch_size * max_tgt_len, 1]

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


def sequence_mask(lengths, max_len=None, dtype=None, device=None) :
    r"""Return a mask tensor representing the first N positions of each cell.
    If ``lengths`` has shape ``[d_1, d_2, ..., d_n]`` the resulting tensor
    ``mask`` has dtype ``dtype`` and shape ``[d_1, d_2, ..., d_n, maxlen]``,
    with
    ```
    mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
    ```
    Examples:
    ```python
    sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                 #  [True,  True,  True, False, False],
                                 #  [True,  True, False, False, False]]
    sequence_mask([[1, 3],[2,0]])  # [[[ True, False, False],
                                   #   [ True,  True,  True]],
                                   #  [[ True,  True, False],
                                   #   [False, False, False]]]
    ```
    Args:
        lengths: integer tensor or list of int, all its values <= max_len.
        max_len: scalar integer tensor, size of last dimension of returned
            tensor. Default is the maximum value in ``lengths``.
        dtype: the desired data type of returned tensor. Default: if None,
            returns :torch:`ByteTensor`.
        device: the desired device of returned tensor. Default: if None, uses
            the current device for the default tensor type.
    Returns:
        A mask tensor of shape :python:`lengths.shape + (max_len,)`, cast to
        specified dtype.
    Raises:
        ValueError: if ``max_len`` is not a scalar.
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    elif device is None:
        device = lengths.device
    lengths: torch.LongTensor
    if max_len is None:
        max_len = torch.max(lengths).item()

    size = lengths.size()
    row_vector = torch.arange(max_len, device=device, dtype=lengths.dtype).view(
        *([1] * len(size)), -1).expand(*size, max_len)
    mask = (row_vector < lengths.unsqueeze(-1)).to(device=device)
    if dtype is not None:
        mask = mask.to(dtype=dtype)

    return mask


def masked_reverse_cumsum(X, lengths, dim):
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])
    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def single_step_PCL_loss(logits, logits_, actions, rewards, seq_lens):
    """
    Single-step unified path consistency learning (PCL). 
    
    See paper https://arxiv.org/pdf/2106.07704.pdf (Eq.15).

    Args:
        logits: [batch_size, tgt_len, vocab_size]
        logits_: [batch_size, tgt_len, vocab_size]
        actions: [batch_size, tgt_len]
        rewards: [batch_size]
        seq_lens: [batch_size]
    
    """
    # calculate policy pi, which equals the advantage function
    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)
    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = logits.logsumexp(dim=-1)
    A = Q - V
    
    # calculate V(s_t+1) + r_t - V(s_t)
    A_ = torch.zeros_like(Q)
    V_ = logits_.logsumexp(dim=-1)
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1]
    
    terminal_V_ = V_[
        torch.arange(seq_lens.shape[0]),
        seq_lens - 1]  # terminal_V_: [batch_size]

    A_[torch.arange(seq_lens.shape[0]),
       seq_lens - 1] = rewards - terminal_V_
    
    raw_losses = F.mse_loss(A, A_, reduction="none")
    return raw_losses


def single_step_PCL_loss_with_seq_rewards(logits, logits_, actions, rewards, seq_lens):

    # calculate policy pi, which equals the advantage function
    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)
    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = logits.logsumexp(dim=-1)
    A = Q - V
    
    # calculate V(s_t+1) + r_t - V(s_t)
    A_ = torch.zeros_like(Q)
    V_ = logits_.logsumexp(dim=-1)
    A_[:, :-1] = V_[:, 1:] - V_[:, :-1] + rewards[:, :-1]
    
    terminal_V_ = V_[
        torch.arange(seq_lens.shape[0]),
        seq_lens - 1]  # terminal_V_: [batch_size]

    terminal_R = rewards[
        torch.arange(seq_lens.shape[0]),
        seq_lens - 1]

    A_[torch.arange(seq_lens.shape[0]),
       seq_lens - 1] = terminal_R - terminal_V_
    
    raw_losses = F.mse_loss(A, A_, reduction="none")
    return raw_losses


def multi_step_PCL_loss(logits, logits_, actions, rewards, seq_lens):
    """
    Multi-step unified path consistency learning (PCL). 
    
    See paper https://arxiv.org/pdf/2106.07704.pdf (Eq.17).

    Args:
        logits: [batch_size, tgt_len, vocab_size]
        logits_: [batch_size, tgt_len, vocab_size]
        actions: [batch_size, tgt_len]
        rewards: [batch_size]
        seq_lens: [batch_size]
    
    """
    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)

    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    # Target outputs
    V_ = logits_.logsumexp(dim=-1)
    A2 = masked_reverse_cumsum(
        A,
        lengths=seq_lens,
        dim=-1)

    raw_losses = F.mse_loss(
        A2, rewards.view(-1, 1) - V_,
        reduction="none")
    return raw_losses


def multi_step_PCL_loss_with_seq_rewards(logits, logits_, actions, rewards, seq_lens):

    if logits.dim() == actions.dim() + 1:
        actions = actions.unsqueeze(-1)

    Q = logits.gather(dim=-1, index=actions).squeeze(-1)
    V = logits.logsumexp(dim=-1)
    A = Q - V

    A2 = masked_reverse_cumsum(
        A,
        lengths=seq_lens,
        dim=-1)

    V_ = logits_.logsumexp(dim=-1)
    R = masked_reverse_cumsum(
        rewards,
        lengths=seq_lens,
        dim=-1)

    raw_losses = F.mse_loss(
        A2, R - V_,
        reduction="none")
    return raw_losses


def mixed_PCL_loss(logits, logits_, actions, rewards, seq_lens, ignore_index=None, reduce=True):
    """
    A mix of single- and multi-step PCL update.

    """
    if rewards.dim() == 1:
        s_pcl = single_step_PCL_loss(logits, logits_, actions, rewards, seq_lens)
        m_pcl = multi_step_PCL_loss(logits, logits_, actions, rewards, seq_lens)

    elif rewards.dim() == 2:
        s_pcl = single_step_PCL_loss_with_seq_rewards(logits, logits_, actions, rewards, seq_lens)
        m_pcl = multi_step_PCL_loss_with_seq_rewards(logits, logits_, actions, rewards, seq_lens)

    else:
        raise Exception("Rewards shape does NOT seems right: {}.".format(rewards.shape))

    raw_losses = (s_pcl + m_pcl) / 2
    assert raw_losses.shape == actions.shape, "Losses shape does not match: {}".format(raw_losses.shape)

    # mask & reduce
    if ignore_index is not None:
        pad_mask = actions.eq(ignore_index)
        raw_losses.masked_fill_(pad_mask, 0.0)

    if reduce:
        raw_losses = raw_losses.mean()
    
    return raw_losses


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, tgt_model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training

        Args:
            model (BARTModel): the model
            tgt_model (BARTModel): the target model
            sample (dict): {
                'id' (Tensor): [batch_size],
                'nsentences' (int): number of samples, 
                'ntokens' (int): total number of tokens?, 
                'net_input': {
                    'src_tokens' (Tensor): [batch_size, max_src_len], 
                    'src_lengths' (Tensor): [batch_size], 
                    'prev_output_tokens' (Tensor): [batch_size, max_tgt_len], 
                }, 
                'target' (Tensor): [batch_size, max_tgt_len]
            }
            net_output (Tensor): [batch_size, max_tgt_len, vocab_size]
        
        """
        net_output = model(**sample["net_input"])
        with torch.no_grad():
            tgt_output = tgt_model(**sample["net_input"])

        loss = self.compute_loss(
            model,
            net_output,
            sample,
            tgt_model=tgt_model,
            tgt_output=tgt_output,
            reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) # log_softmax: [batch_size, max_tgt_len, vocab_size]
        target = model.get_targets(sample, net_output) # target: [batch_size, max_tgt_len]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_probs(self, model, net_output):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        if self.ignore_prefix_size > 0:
            raise Exception("Does not support ignore prefix for now.")
        return probs.view(-1, probs.size(-1))

    def compute_loss(
            self,
            model,
            net_output,
            sample,
            tgt_model=None,
            tgt_output=None,
            reduce=True
        ):
        """
        Args:
            net_output (tuple):
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        target = model.get_targets(sample, net_output)  # target: [batch_size, max_tgt_len]

        p_mle = None
        if sample.get('p_mle', None) is not None:
            p_mle = sample['p_mle']
            assert target.size() == p_mle.size(), "Target size: {}; P_MLE size: {}.".format(target.size(), p_mle.size())

        # rewards = p_mle.sum(dim=-1)
        rewards = p_mle
        if rewards.dtype != net_output[0].dtype and net_output[0].dtype == torch.float16:
            rewards = rewards.half()

        loss = mixed_PCL_loss(
            net_output[0],
            tgt_output[0],
            target,
            rewards,
            sample['tgt_lengths'],
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
