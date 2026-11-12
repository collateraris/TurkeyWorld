"""Discarded H16/H96/H450 pure-XY GPU probes for the loss-gated curriculum."""
from .probe_rssm_drift_only import main


if __name__=='__main__':
    main(loss_gated=True)
