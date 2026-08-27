# M1 one-epoch smoke comparison

This directory records the conservative one-epoch feasibility gate using a
`1e-5` learning rate and 10:100 energy/force weighting. It showed consistent,
small held-out force improvements and justified the five-epoch experiment in
`../m1_5epoch/`.

The model itself remains Git-ignored. `comparison.json` records its SHA-256 hash
and compares it with M0 on identical structures.
