"""
The term for the subdivision of an island in Hawaii into multiple agricultural zones is "ahupuaʻa." 
Ahupuaʻa is a traditional Hawaiian system of land division that typically extends from 
the mountains to the sea, encompassing a variety of ecosystems. This system allowed for the 
sustainable management of resources across different ecological zones. Each ahupuaʻa contained 
nearly all the resources the Hawaiian community living within its boundaries would need, 
which minimized the need for long-distance travel to gather resources. These zones included 
upland forests, agricultural lands, and fishing grounds, among others.

The methods here mostly use Polars and numpy, so moving to Rust is not that necessary so far.
"""

import polars
import numpy as np
from typing import Optional

class Partition:
    def __init__(self) -> None:
        pass

class MetricsPartition(Partition):
    def __init__(self, ts: np.ndarray, values: np.ndarray, metadata = Optional[dict]) -> None:
        super().__init__()
        self.ts = ts
        self.values = values
        self.metadata = metadata
    
    def __str__(self):
        return f"metrics segment: " + str(self.compute_stats())

    def compute_stats(self) -> None:
        min_val, max_val, average_val = np.min(self.values), np.max(self.values), np.mean(self.values)
        return {"num_measurements": len(self.ts), 
                "min_ts": str(np.min(self.ts)), 
                "max_ts": str(np.max(self.ts)), 
                "time_span": str(np.max(self.ts) - np.min(self.ts)),
                "min_val": min_val, 
                "max_val": max_val, 
                "average_val": average_val}


def check_time_series(ts: np.ndarray, values: np.ndarray):
    if len(ts) != len(values):
        raise ValueError("ts and values must have the same length")
    try:
        ts = np.array(ts, dtype = np.datetime64)
    except:
        raise ValueError("ts must be a numpy array of datetime64 or convertible to it")
    return ts

def partition_sessionize(ts: np.ndarray, values: np.ndarray, gap = 20):
    
    ts = check_time_series(ts, values)
    
    nnz = np.nonzero(values)
    breaks = np.diff(nnz[0]) > gap
    assgnmts = np.hstack([[0], np.cumsum(breaks)])
    split_indices = np.argwhere(assgnmts[:-1] != assgnmts[1:]).flatten() + 1
    ts = ts[nnz]
    values = values[nnz]
    return [MetricsPartition(ts_split, values_split) for ts_split, values_split in 
            zip(np.split(ts, split_indices),  np.split(values, split_indices))]
   

def partition_periodic(ts: np.ndarray, values: np.ndarray, period = None):
    
    ts = check_time_series(ts, values)
    
    confidence_metric = 100 if period is not None else None 
    if period is None:
        dft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values), d=1)
        magnitudes = np.abs(dft)
        magnitudes_no_zero = magnitudes[1:]
        freqs_no_zero = freqs[1:]
        max_magnitude_idx = np.argmax(magnitudes_no_zero)
        dominant_freq = freqs_no_zero[max_magnitude_idx]
        dominant_magnitude = magnitudes_no_zero[max_magnitude_idx]
        most_likely_period = 1 / dominant_freq
        
        total_magnitude = np.sum(magnitudes_no_zero)
        confidence_metric = dominant_magnitude / total_magnitude
        period = most_likely_period
    
    offsets = np.arange(int(period), len(values), int(period))
    return [MetricsPartition(ts_split, values_split) for ts_split, values_split in 
            zip(np.split(ts, offsets),  np.split(values, offsets))]