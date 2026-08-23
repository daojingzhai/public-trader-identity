# Public Trader Identity

Github repo for

> Daojing Zhai (2026), *Public Trader Identity: Adverse Selection and Return Predictability*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7234658)

| folder | what it is |
|---|---|
| `dec2025_reconstruction/` | Appendix C: recovering timing and ordering from the public Hyperliquid December 2025 data (Albers et al., Zenodo) and replaying the order book into the gate-valid BBO tape, with the audit tables |

## Release status

This is an early public release accompanying a working paper. It contains the core
Appendix C reconstruction code, released to make clear how the public December 2025 data
were used. The remaining estimation code and a compact replication package will be added
after publication.

## December 2025 data

The December 2025 data are publicly available from Albers et al. at
[Zenodo DOI 10.5281/zenodo.18184441](https://doi.org/10.5281/zenodo.18184441). The public
book diffs do not contain the consensus block number, block timestamp, or oracle stream. The
code constructs a conservative `availableTime` from the public order-status and trade
timestamps while preserving book-diff file order. Output `blockNumber` values are synthetic
replay counters for atomic update groups, not recovered Hyperliquid chain blocks.

The July 2026 raw node archive and the large derived panels are not stored on GitHub;
data-access documentation will accompany the complete release.

## Use and citation

If you use or adapt the code, please cite the paper and the specific repository release.
Substantial extensions using this reconstruction would benefit from coordination with the
author.

The MIT license applies to the code only. The underlying public data remain subject to their
source license and attribution requirements.
