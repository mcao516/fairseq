# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import pickle

from datetime import datetime
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

UNK_ID = 3
Alpha = 1.0

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, mask=None):
    """
    Args:
        lprobs (tensor): [bs * tgt_length, vocab_size]
        target (tensor): [bs * tgt_length]
        mask (tensor): [bs * tgt_length]
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)  # nll_loss: [bs * tgt_length, 1]
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # smooth_loss: [bs * tgt_length, 1]

    # if mask is not None:
    #     nll_loss.masked_fill_((1 - mask).bool(), 0.)
    #     smooth_loss.masked_fill_((1 - mask).bool(), 0.)

    if ignore_index is not None:  # 1
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # ====== calculate abstention loss ======
    lprob_unk = lprobs[:, UNK_ID]  # lprob_unk: [bs * tgt_length]
    prob_unk = torch.exp(lprob_unk)  # prob_unk: [bs * tgt_length]
    prob_unk_reduced = (1. - prob_unk)

    a = prob_unk_reduced.masked_fill_(mask.eq(0), 1.0).unsqueeze(-1)
    b = (prob_unk_reduced - Alpha) * torch.log(prob_unk_reduced) * mask
    b = b.unsqueeze(-1)

    # for analysis
    nll_loss_regularized = a * nll_loss
    regularizer = b

    nll_loss = a * nll_loss + b
    smooth_loss = a * smooth_loss + b
    # =======================================

    if reduce:  # True
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    
    eps_i = epsilon / lprobs.size(-1)  # 0.1
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss, nll_loss_regularized, regularizer


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

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, _, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        # ==================================================================================================

        ids = sample['id'].tolist()
        i = min(ids)
        if i < 30 and model.training:
            with torch.no_grad():
                _, nll_token_loss, nll_loss_regularized, regularizer = self.compute_loss(model, net_output, sample, reduce=False)

            data_to_save = {
                'sample': sample,
                'token_loss': nll_token_loss.detach(),
                'sentence_loss': nll_loss.detach(),
                'token_loss_regularized': nll_loss_regularized.detach(),
                'regularizer': regularizer.detach()
            }

            path = '/home/mcao610/fairseq/loss_analysis/{}.{:%Y%m%d_%H%M%S}.obj'.format(i, datetime.now())
            torch.save(data_to_save, path)
        
        # ==================================================================================================

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
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
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)

        # ========= loss abstention =========
        mask = None
        if sample.get('mask', None) is not None:
            mask = sample['mask'].view(-1)
            assert target.size() == mask.size(), "Target size: {}; Mask size: {}.".format(target.size(), mask.size())
        assert mask is not None
        # ===================================

        loss, nll_loss, nll_loss_regularized, regularizer = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            mask=mask
        )
        return loss, nll_loss, nll_loss_regularized, regularizer

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
