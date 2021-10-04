# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def transpose_batch_time(inputs):
    r"""Transposes inputs between time-major and batch-major.
    """
    return inputs.transpose(0, 1)


def mask_sequences(
    sequence,
    sequence_length,
    dtype=None,
    time_major=False
):
    if not torch.is_tensor(sequence):
        sequence = torch.tensor(sequence, dtype=dtype)
    sequence: torch.Tensor

    rank = sequence.dim()
    if rank < 2:
        raise ValueError("`sequence` must be 2D or higher order.")

    if time_major:
        sequence = transpose_batch_time(sequence)
    max_time = sequence.size(1)
    if dtype is None:
        dtype = sequence.dtype
    mask = sequence_mask(sequence_length, max_time, dtype=dtype)
    mask = mask.view(*mask.size(), *([1] * (rank - 2)))
    sequence = sequence * mask
    if time_major:
        sequence = transpose_batch_time(sequence)

    return sequence


def reduce_batch_time(
    sequence,
    sequence_length,
    average_across_batch=True,
    average_across_timesteps=False,
    sum_over_batch=False,
    sum_over_timesteps=True
):
    if average_across_timesteps and sum_over_timesteps:
        raise ValueError("Only one of `average_across_timesteps` and "
                         "`sum_over_timesteps` can be set.")
    if average_across_batch and sum_over_batch:
        raise ValueError("Only one of `average_across_batch` and "
                         "`sum_over_batch` can be set.")

    if sum_over_timesteps:
        sequence = torch.sum(sequence, dim=1)
    elif average_across_timesteps:
        if sequence_length is None:
            sequence = torch.mean(sequence, dim=1)
        else:
            sequence = (torch.sum(sequence, dim=1).float() /
                        sequence_length.float())

    if sum_over_batch:
        sequence = torch.sum(sequence, dim=0)
    elif average_across_batch:
        sequence = torch.mean(sequence, dim=0)

    return sequence


def mask_and_reduce(
    sequence,
    sequence_length,
    average_across_batch=True,
    average_across_timesteps=False,
    sum_over_batch=False,
    sum_over_timesteps=True,
):
    sequence = mask_sequences(sequence,
                              sequence_length,
                              dtype=None,
                              time_major=False)

    return reduce_batch_time(sequence,
                             sequence_length,
                             average_across_batch,
                             average_across_timesteps,
                             sum_over_batch,
                             sum_over_timesteps)


def label_smoothed_nll_loss(
    lprobs,
    target,
    epsilon,
    tgt_probs,
    mle_probs,
    seq_lens,
    ignore_index=None,
    reduce=True,
    gamma=1.0,
    min_pi_theta=0.0
):
    """
    Args:
        lprobs (Tensor): [batch_size * max_tgt_len, vocab_size]
        target (Tensor): [batch_size * max_tgt_len]
        tgt_probs (Tensor): [batch_size * max_tgt_len, vocab_size]
        mle_probs (Tensor): [batch_size * max_tgt_len]
        seq_lens (Tensor): [batch_size]
    """
    # if str(lprobs.device) == 'cuda:0':
    #     # print(torch.exp(lprobs[0][:10]))
    #     # print(target[0])
    #     print(tgt_probs[0][:10])
    #     # print(mle_probs[0])
    #     # print(seq_lens)

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1) # [batch_size * max_tgt_len, 1]
    nll_loss = -lprobs.gather(dim=-1, index=target) # [batch_size * max_tgt_len, 1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True) # [batch_size * max_tgt_len, 1]

    pi_theta = tgt_probs.gather(dim=-1, index=target)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        pi_theta.masked_fill_(pad_mask, 1.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # calculate rewards
    with torch.no_grad():
        pi_theta = torch.clamp(pi_theta, min=min_pi_theta, max=1.0).detach()
        Q = discounted_future_sum(mle_probs, seq_lens, num_steps=5, gamma=gamma)
        Q = Q.view(-1).unsqueeze(-1).detach()
    assert pi_theta.shape == Q.shape == nll_loss.shape
    nll_loss = (pi_theta * Q) * nll_loss

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
    """
    Args:
        X (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        dim (int): -1
        gamma (float): the discount factor
    
    """
    masked_X = X * sequence_mask(lengths, max_len=X.shape[1])
    return (masked_X
            .flip(dims=[dim])
            .cumsum(dim=dim)
            .flip(dims=[dim]))


def discounted_future_sum(values, lengths, num_steps=None, gamma=1.0):
    """
    Args:
        values (Tensor): [batch_size, max_tgt_len]
        lengths (Tensor): [batch_size]
        num_steps (int): A positive integer number of future steps to sum
        gamma (float): A float discount value
    
    """
    assert values.dim() == 2
    
    batch_size, total_steps = values.shape
    values = values * sequence_mask(lengths, max_len=values.shape[1])

    num_steps = total_steps if num_steps is None else num_steps
    num_steps = min(num_steps, total_steps)
    
    padding = torch.zeros([batch_size, num_steps - 1]).to(values)
    padded_values = torch.cat([values, padding], 1)
    discount_filter = gamma ** torch.arange(num_steps).to(values).reshape(1, 1, -1)

    output = F.conv1d(padded_values.unsqueeze(-2), discount_filter).squeeze(1)
    return output


def get_reward_shaping_func(
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float
):
    def _shaping_func(rewards):
        percentile = (rewards - old_min) / (old_max - old_min)
        return percentile * (new_max - new_min) + new_min

    return _shaping_func


@register_criterion("label_smoothed_cross_entropy")
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        reward_shaping=False,
        old_r_min=0.,
        old_r_max=1.0,
        new_r_min=-0.5,
        new_r_max=0.5,
        gamma=1.0,
        min_pi_theta=1.0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.gamma = gamma
        self.min_pi_theta = min_pi_theta

        if reward_shaping:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=old_r_min,
                old_max=old_r_max,
                new_min=new_r_min,
                new_max=new_r_max)
        else:
            self._reward_shaping_func = lambda r: r

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
        parser.add_argument('--reward-shaping', action='store_true',
                            help='Whether use reward shaping')
        parser.add_argument('--old-r-min', default=0., type=float,
                            help='Original minimum reward value')
        parser.add_argument('--old-r-max', default=1.0, type=float,
                            help='Original maximum reward value')
        parser.add_argument('--new-r-min', default=-0.5, type=float,
                            help='Minimum reward value after reshaping')
        parser.add_argument('--new-r-max', default=0.5, type=float,
                            help='Maximum reward value after reshaping')
        parser.add_argument('--gamma', default=1.0, type=float)
        parser.add_argument('--min_pi_theta', default=0.0, type=float,
                            help='Minimum Pi_theta')
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
                'rewards' (Tensor): [batch_size, max_tgt_len] or [batch_size]
            }
            net_output (Tensor): [batch_size, max_tgt_len, vocab_size]
        
        """
        net_output = model(**sample["net_input"])
        tgt_model.eval()
        with torch.no_grad():
            tgt_output = tgt_model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(
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

        rewards = None
        if sample.get('rewards', None) is not None:
            rewards = sample['rewards']
            assert rewards.shape[0] == target.shape[0] and \
                (rewards.dim() == 1 or rewards.shape[1] == target.shape[1]), \
                "Target size: {}; rewards size: {}.".format(target.size(), rewards.size())

        rewards = self._reward_shaping_func(rewards)
        if rewards.dtype != net_output[0].dtype and net_output[0].dtype == torch.float16:
            rewards = rewards.half()

        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        tgt_probs = self.get_probs(tgt_model, tgt_output)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            tgt_probs,
            rewards,
            sample['tgt_lengths'],
            ignore_index=self.padding_idx,
            reduce=reduce,
            gamma=self.gamma,
            min_pi_theta=self.min_pi_theta
        )
        return loss, nll_loss

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
