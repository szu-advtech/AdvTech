#!/usr/bin/env python

import torch.nn as nn


class TCNN(nn.Module):
    LENGTH_S = 20
    LENGTH_M = 100
    LENGTH_L = 300

    def __init__(self, features_in, features_out, p_dropout=0.0, size="M"):
        """
        Temporal Convolutional Neural Network

        features_in  :: size of alphabet (nrows of input matrix, default 37)
        features_out :: size of final layer
        size         :: 'S' small, 'M' medium, or 'L' large network

        """
        super().__init__()
        self.module_dict = nn.ModuleDict()

        cnn_out_dim = 0
        if size == "S":
            self.minimal_length = self.LENGTH_S
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(2),

                nn.Conv1d(256, 512, kernel_size=2, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            cnn_out_dim = 512
        elif size == "M":
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 1024, kernel_size=3, padding=0),
                nn.BatchNorm1d(1024),
                nn.ReLU()
            )
            cnn_out_dim = 1024
        elif size == "L":
            self.minimal_length = self.LENGTH_L
            self.conv = nn.Sequential(
                nn.Conv1d(features_in, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),

                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(3),

                nn.Conv1d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Conv1d(1024, 2048, kernel_size=3, padding=0),
                nn.BatchNorm1d(2048),
                nn.ReLU()
            )
            cnn_out_dim = 2048
        self.module_dict['conv'] = self.conv

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, cnn_out_dim),
            nn.PReLU(),
            nn.Dropout(p=p_dropout),

            nn.Linear(cnn_out_dim, features_out)
        )
        self.module_dict['fc'] = self.fc

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)

        return X


def out_dim(seq_length, kernel_size, padding=0, stride=1, dilation=1):
    return (seq_length+2*padding-dilation*(kernel_size-1)-1)//stride+1
