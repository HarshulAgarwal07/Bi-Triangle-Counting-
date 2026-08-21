import random, sys

# usage: gen_graph.py out.txt nL nU nE seed [neg_prob]
out, nL, nU, nE, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
neg = float(sys.argv[6]) if len(sys.argv) > 6 else 0.35

random.seed(seed)
edges = {}
# skewed degrees: pick endpoints with a power-law-ish bias so some vertices are hot
def pick(n):
    # bias toward low indices
    return min(n - 1, int(n * (random.random() ** 2.2)))

attempts = 0
while len(edges) < nE and attempts < nE * 40:
    attempts += 1
    l, u = pick(nL), pick(nU)
    if (l, u) in edges:
        continue
    edges[(l, u)] = 1 if random.random() < neg else 0

with open(out, "w", newline="\n") as f:
    f.write("%d %d %d\n" % (nL, nU, len(edges)))
    for (l, u), s in edges.items():
        f.write("%d %d %d\n" % (l, u, s))

print("wrote %s: nL=%d nU=%d nE=%d" % (out, nL, nU, len(edges)))
