# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

_EPSILON = 1e-6


class Clipper:
    def __init__(
        self,
        should_clip_scores: bool,
        max_score_norm: None,
    ):
        self._should_clip_scores = should_clip_scores
        self.max_score_norm = max_score_norm

    @property
    def should_clip_scores(self) -> bool:
        return self._should_clip_scores

    def clip_scores(self, scores) -> torch.Tensor:
        if self.should_clip_scores:
            score_norms = torch.sqrt((scores**2).sum(-1))
            # print(score_norms)
            clip_coefficient = torch.clamp(
                self.max_score_norm / (score_norms + _EPSILON), max=1
            )
            return scores * clip_coefficient[:, None]
        else:
            return scores


class Clipper1d(Clipper):
    def clip_scores(self, scores) -> torch.Tensor:
        if self.should_clip_scores:
            score_norms = scores.abs()
            # print(score_norms)
            clip_coefficient = torch.clamp(
                self.max_score_norm / (score_norms + _EPSILON), max=1
            )
            return scores * clip_coefficient
        else:
            return scores
