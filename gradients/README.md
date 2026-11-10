# Diffusion gradient tables

The 6-, 9-, 15- and 22-direction tables are copied without modification from the original study files. Each includes one leading b0 entry, followed by DW entries at b = 1000 s/mm². The b-vector files use one vector per row (`N × 3`). The readers also accept FSL `3 × N` files.

The small rounding deviations from unit length in the stored tables are retained. Code normalizes nonzero vectors when reading them. These are the study protocol tables; their inclusion does not mean that a separate SH-DTI++ model was trained for every direction count. The archived main training configuration uses 15 DW directions.
