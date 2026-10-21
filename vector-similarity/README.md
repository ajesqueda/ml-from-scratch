# Vector norms, distances and similarity

The small set of vector operations behind k-nearest neighbors, k-means, embedding
search and recommenders. Each is written as a plain Python loop that spells out the
formula and as a vectorized NumPy version, and the tests check that they agree.

> Concept from BU DX601 (Math for Data Science), Week 8: vectors and matrices.

## What's inside (`vectors.py`)

| Function | Formula | Use it for |
|----------|---------|-----------|
| `l1_norm` | Σ\|vᵢ\| | Grid / "Manhattan" distance, sparse models |
| `l2_norm` | √Σvᵢ² | Straight-line length |
| `max_norm` | max\|vᵢ\| | Worst single coordinate |
| `unit_vector` | v / ‖v‖ | Direction only (magnitude × direction split) |
| `euclidean_distance` | ‖a − b‖ | "How far apart?" |
| `cosine_similarity` | a·b / (‖a‖‖b‖) | "Pointing the same way?", ignoring length |
| `nearest_neighbor` | argminᵢ ‖q − pᵢ‖ | 1-NN search |
| `np_pairwise_distances` | broadcasting `(q,1,d) − (1,p,d)` | Every distance at once, no loop |

## Run it

```bash
pip install -r requirements-dev.txt
python vector-similarity/vectors.py
pytest vector-similarity
```

```
houses (sqft, beds): [[2000.0, 2.0], [2050.0, 5.0], [2300.0, 3.0]]  query: [2040.0, 3.0]
nearest without scaling: house 1 [2050.0, 5.0]  (sqft dominates)
nearest after scaling:   house 0 [2000.0, 2.0]

short vs long doc, same topic: Euclidean = 20.1, cosine = 1.00
```

## Two takeaways

1. **Units decide your neighbors.** Square feet are about 1000× bigger than bedroom
   counts, so unscaled distance effectively ignores bedrooms. After standardizing each
   column to mean 0 and std 1, the 3-bedroom query matches the 2-bedroom house over the
   5-bedroom one. Scale features before any distance-based model.
2. **Cosine ignores length; Euclidean doesn't.** Two documents about the same topic,
   one 10× longer, are far apart in Euclidean distance but have cosine similarity 1.
   That's why embedding search uses cosine.

The tests also check the norm ordering `max ≤ L2 ≤ L1` on random vectors, and that
cosine similarity equals the cosine of the actual angle between two vectors.
