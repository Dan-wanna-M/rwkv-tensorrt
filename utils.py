########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import sys
import time
import random
import os
import re
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

time_slot = {}
time_ref = time.time_ns()


class SAMPLER():
    def __init__(self, sample, temp, top_p, tau, count_penalty, presence_penalty, penalty_range):
        if sample == 'nucleus':
            self.sample = self.sample_nucleus
        elif sample == 'typical':
            self.sample = self.sample_typical
        else:
            raise RuntimeError("\"sample\" must be \"nucleus\" or \"typical\"")

        self.temp = temp
        self.top_p = top_p
        self.top_k = 0
        self.tau = tau
        self.count_penalty = count_penalty
        self.presence_penalty = presence_penalty
        self.penalty_range = penalty_range

    def __str__(self) -> str:
        method = "Nucleus" if self.sample == self.sample_nucleus else "Typical"
        return '''|{:^30}|{:^10}|
|------------------------------|----------|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
'''.format("Sampler Params", "Values",
           "Method", method,
           "Temperature", self.temp,
           "Top P", self.top_p,
           "Tau", self.tau,
           "Count Penalty", self.count_penalty,
           "Presence Penalty", self.presence_penalty,
           "Penalty Range", self.penalty_range)


    def sample_nucleus(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(
                cumulative_probs > self.top_p)])
            probs[probs < cutoff] = 0
            if self.top_k < len(probs) and self.top_k > 0:
                probs[sorted_ids[:-self.top_k]] = 0
            if self.temp != 1.0:
                probs = probs ** (1.0 / self.temp)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(
                cumulative_probs > self.top_p)])
            probs[probs < cutoff] = 0
            if self.top_k < len(probs) and self.top_k > 0:
                probs[sorted_ids[:-self.top_k]] = 0
            if self.temp != 1.0:
                probs = probs ** (1.0 / self.temp)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def sample_typical(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        entropy = torch.nansum(logits * probs, dim=-1, keepdim=True)
        logits = torch.abs(logits - entropy)
        sorted_ids = torch.argsort(logits)
        sorted_logits = logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < self.tau)
        probs[logits > sorted_logits[cutoff]] = 0
        if self.temp != 1.0:
            probs = probs ** (1.0 / self.temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
