#!/usr/bin/env python
# encoding: utf-8
# author: Hu Xin
# email: qzlyhx@gmail.com
# time: 2022.8.10
# description:
#   This file contains definition of Variational Sampler


import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.config import getDevice

class Sampler(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Sampler, self).__init__()
        device = getDevice()
        self.fcToMean = nn.Linear(inputSize, outputSize, device=device)
        self.fcToLog = nn.Linear(inputSize, outputSize, device=device)
        self.outMap = nn.Linear(outputSize, outputSize, device=device)
    
    def getParameters(self):
        paraList = [{'params': self.fcToMean.parameters()}, {'params': self.fcToLog.parameters()}, {'params': self.outMap.parameters()}]
        return paraList

    def initParameters(self, initFunc):
        initFunc(self.fcToMean.weight, std=0.03)
        initFunc(self.fcToLog.weight, std=0.03)
        initFunc(self.fcToMean.bias, std=0.03)
        initFunc(self.fcToLog.bias, std=0.03)
        initFunc(self.outMap.weight, std=0.03)
        initFunc(self.outMap.bias, std=0.03)

    def forward(self, x):
        # convert feature x to mean and log
        mean = self.fcToMean(x)
        logVar = self.fcToLog(x)
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        sampled = eps.mul(std).add_(mean)
        sampled = self.outMap(sampled)
        # calculate KL Divergence
        kld = - 0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())

        return sampled, kld
    
    def mix2(self, x0, x1, rate):
        mean0 = self.fcToMean(x0)
        logVar0 = self.fcToLog(x0)
        mean1 = self.fcToMean(x1)
        logVar1 = self.fcToLog(x1)
        mean = mean0*(1-rate) + mean1*rate
        logVar = logVar0*(1-rate) + logVar1*rate
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        sampled = eps.mul(std).add_(mean)
        sampled = self.outMap(sampled)

        return sampled



