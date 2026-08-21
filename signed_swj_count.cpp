// signed_swj_count.cpp
//
// Sequential SIGNED SWJ-Count: counts balanced and unbalanced bi-triangles
// in a signed bipartite graph in one pass, using the super-wedge join
// (SWJ-Count) method from Yang et al., VLDB 2021, "Efficient Bi-triangle
// Counting for Large Bipartite Networks" (Algorithm 2), extended with
// sign-parity bucketing (BB2K-style, Mekala et al. 2026) so that balance
// is determined during counting rather than via a second enumeration pass.
//
// -----------------------------------------------------------------------
// INPUT FORMAT (plain text):
//
//   nL nU nE
//   l1 u1 s1
//   l2 u2 s2
//   ...
//   lnE unE snE
//
//   - nL : number of vertices in the lower layer L, indices [0, nL)
//   - nU : number of vertices in the upper layer U, indices [0, nU)
//   - nE : number of edges
//   - each edge line: l (L-index), u (U-index), s (0 = positive, 1 = negative)
//
// No duplicate edges, no self loops (bipartite graphs can't have those
// anyway). Multi-edges with conflicting signs are undefined behaviour --
// dedupe before feeding this program.
// -----------------------------------------------------------------------
//
// USAGE:
//   g++ -O2 -std=c++17 -o signed_swj_count signed_swj_count.cpp
//   ./signed_swj_count graph.txt              // run SWJ-Count only
//   ./signed_swj_count graph.txt --verify      // also run brute-force
//                                               // ground truth and diff
//
// The --verify brute force is O(huge) -- only use it on small graphs
// (roughly |L| up to a few hundred, depending on density).
//
// Requires GCC or Clang: this file uses __int128 (see the accumulator
// note below) and <bits/stdc++.h>, neither of which MSVC provides.
//
// -----------------------------------------------------------------------
// ALGORITHM NOTES (why this is correct -- see derivation in chat history)
//
// A bi-triangle (6-cycle) is a "swj-unit": two edge-disjoint super-wedges
// (3-edge paths) s1, s2 sharing both endpoints. Since s1, s2 are exactly
// the 6-cycle's 6 edges split into two disjoint halves:
//
//     #negative edges in bi-triangle = #negative(s1) + #negative(s2)
//
// so the bi-triangle is balanced (even #negative edges) iff
// weight(s1) == weight(s2), where weight(path) = XOR of its edge signs.
//
// This same fact applies recursively inside every acyclic-correction term
// (Type-L, Type-U, self-intersection): in each case the "extra" edge is
// literally reused by both halves being compared, so its sign
// self-cancels under XOR, and the correction's balanced/unbalanced split
// depends only on the relative parity of the two wedges involved -- NOT
// on the sign of whatever extension edge is being counted. This is why
// every scalar count in the unsigned algorithm can be replaced by an
// (even, odd) pair, and every C(n,2) becomes:
//
//     balanced   contribution: C(even,2) + C(odd,2)
//     unbalanced contribution: even * odd
//
// with all other multiplicative factors (degrees, extension-edge counts)
// carried over completely unchanged from the unsigned algorithm.
// -----------------------------------------------------------------------

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// =========================================================================
// 128-bit accumulator support.
//
// The final counts do NOT fit in a signed 64-bit integer on real data:
// ll tops out at ~9.2e18, while the source paper's own published
// bi-triangle counts already exceed that on medium datasets (Reuters is
// ~2.46e19). Worse, the intermediate raw counts Lam/Bar are ~3x the
// final answer (each bi-triangle is counted three times before the
// divide-by-3), so they blow past ll even earlier and the overflow is
// silent -- it shows up as a nonsensical or negative "answer", not a
// crash.
//
// So every accumulator, every C(n,2), and every product of two counts is
// __int128. The per-vertex maps (S, H, T) stay ll: those hold local
// wedge/super-wedge counts bounded by the neighbourhood size, which
// cannot approach 9.2e18. It is only the global sums and the pairwise
// products fed into them that need the extra width.
// =========================================================================
typedef __int128 i128;

static string i128_to_string(i128 n) {
    if (n == 0) return "0";
    bool neg = n < 0;
    unsigned __int128 un = neg ? (unsigned __int128)(-n) : (unsigned __int128)n;
    string s;
    while (un > 0) {
        s += char('0' + (int)(un % 10));
        un /= 10;
    }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    return s;
}

// =========================================================================
// Graph storage
// =========================================================================
struct SignedBiGraph {
    int nL = 0, nU = 0;
    vector<vector<int>> adjL;      // adjL[l] = sorted list of u-indices
    vector<vector<int>> adjU;      // adjU[u] = sorted list of l-indices
    unordered_map<ll, int> signOf; // key = l * (nU+1) + u  ->  sign (0/1)

    inline ll key(int l, int u) const { return (ll)l * (nU + 1) + u; }

    inline int sign(int l, int u) const {
        auto it = signOf.find(key(l, u));
        assert(it != signOf.end() && "sign() called on a non-existent edge");
        return it->second;
    }

    inline int degL(int l) const { return (int)adjL[l].size(); }
    inline int degU(int u) const { return (int)adjU[u].size(); }
};

bool loadGraph(const char* path, SignedBiGraph& G) {
    ifstream fin(path);
    if (!fin) {
        cerr << "ERROR: cannot open input file: " << path << "\n";
        return false;
    }

    int nL, nU;
    ll nE;
    fin >> nL >> nU >> nE;
    if (!fin) {
        cerr << "ERROR: malformed header (expected: nL nU nE)\n";
        return false;
    }
    if (nL < 0 || nU < 0 || nE < 0) {
        cerr << "ERROR: negative sizes in header\n";
        return false;
    }

    G.nL = nL;
    G.nU = nU;
    G.adjL.assign(nL, {});
    G.adjU.assign(nU, {});
    G.signOf.reserve((size_t)nE * 2 + 16);

    for (ll i = 0; i < nE; i++) {
        int l, u, s;
        fin >> l >> u >> s;
        if (!fin) {
            cerr << "ERROR: malformed edge line index " << i << "\n";
            return false;
        }
        if (l < 0 || l >= nL || u < 0 || u >= nU) {
            cerr << "ERROR: edge index out of range at line " << i
                 << " (l=" << l << ", u=" << u << ")\n";
            return false;
        }
        if (s != 0 && s != 1) {
            cerr << "ERROR: sign must be 0 or 1, got " << s
                 << " at line " << i << "\n";
            return false;
        }
        G.adjL[l].push_back(u);
        G.adjU[u].push_back(l);
        G.signOf[G.key(l, u)] = s;
    }

    for (auto& v : G.adjL) sort(v.begin(), v.end());
    for (auto& v : G.adjU) sort(v.begin(), v.end());

    return true;
}

// =========================================================================
// Signed SWJ-Count
// =========================================================================
struct SwjResult {
    i128 balanced = 0;
    i128 unbalanced = 0;
    i128 total() const { return balanced + unbalanced; }
};

// C(n,2) computed in 128-bit: n is a local count (fits in ll), but n*(n-1)
// does not -- n ~ 6e9 already overflows a 64-bit product.
static inline i128 C2(ll n) { return n > 1 ? (i128)n * (n - 1) / 2 : (i128)0; }

SwjResult signedSwjCount(const SignedBiGraph& G) {
    i128 Lam_bal = 0, Lam_unb = 0; // raw swj-unit counts (valid + both acyclic types mixed in)
    i128 Bar_bal = 0, Bar_unb = 0; // acyclic corrections (Type-L + Type-U)

    // =====================================================================
    // PASS 1: anchor at every l in L.
    //   - build S[x] = (even,odd) wedge counts l -> x   (x in L, 2-hop)
    //   - Type-L acyclic correction, using S
    //   - build H[y] = (even,odd) super-wedge counts l -> y (y in U, 3-hop)
    //   - self-intersection fix on H
    //   - accumulate Lam_bal / Lam_unb from H
    // =====================================================================
    for (int l = 0; l < G.nL; l++) {

        // ---- build S: 2-hop neighbours of l (x in L), split by wedge weight ----
        // wedge l - u - x has weight = sign(l,u) XOR sign(u,x)
        unordered_map<int, pair<ll, ll>> S; // x -> (evenWedges, oddWedges)
        S.reserve((size_t)G.degL(l) * 4 + 4);

        for (int u : G.adjL[l]) {
            int s_lu = G.sign(l, u);
            for (int x : G.adjU[u]) {
                if (x == l) continue;
                int s_ux = G.sign(x, u);
                int w = s_lu ^ s_ux;
                auto& p = S[x];
                if (w == 0) p.first++; else p.second++;
            }
        }

        // ---- Type-L acyclic correction ----
        // Two wedges l-u1-x, l-u2-x (same x) extended by an edge (x,t),
        // t != {u1,u2}, giving roughly d(x)-2 choices of t.
        // weight(s1) XOR weight(s2) = wedgeWeight1 XOR wedgeWeight2
        // (the shared extension edge (x,t) cancels via XOR self-cancellation).
        for (auto& kv : S) {
            int x = kv.first;
            ll se = kv.second.first, so = kv.second.second;
            ll ext = (ll)G.degL(x) - 2;
            if (ext > 0) {
                Bar_bal += (C2(se) + C2(so)) * ext;
                Bar_unb += ((i128)se * so) * ext;
            }
        }

        // ---- build H: 3-hop neighbours of l (y in U), split by super-wedge weight ----
        // super-wedge l-u-x-y has weight = wedgeWeight(l,u,x) XOR sign(x,y)
        unordered_map<int, pair<ll, ll>> H; // y -> (evenSW, oddSW)
        H.reserve((size_t)G.degL(l) * 8 + 8);

        for (auto& kv : S) {
            int x = kv.first;
            ll se = kv.second.first, so = kv.second.second;
            for (int y : G.adjL[x]) {
                int s_xy = G.sign(x, y);
                auto& h = H[y];
                if (s_xy == 0) { h.first += se; h.second += so; }
                else           { h.first += so; h.second += se; }
            }
        }

        // ---- self-intersection fix ----
        // For y in N(l): the walk l-u-x-y degenerates (u == y, path
        // revisits y) exactly d(y)-1 times, once for each x in N(y)\{l}.
        // Every such spurious contribution has weight == sign(l,y)
        // (constant across all x -- the reused edge (x,y) cancels via
        // XOR self-cancellation, leaving only sign(l,y)).
        for (int y : G.adjL[l]) {
            ll corr = (ll)G.degU(y) - 1;
            if (corr <= 0) continue;
            auto it = H.find(y);
            if (it == H.end()) continue; // no 3-hop paths reached y at all
            int s_ly = G.sign(l, y);
            if (s_ly == 0) {
                it->second.first -= corr;
                if (it->second.first < 0) {
                    cerr << "WARNING: negative even-count after self-intersection "
                         << "fix at l=" << l << " y=" << y
                         << " (indicates a bug -- please report)\n";
                }
            } else {
                it->second.second -= corr;
                if (it->second.second < 0) {
                    cerr << "WARNING: negative odd-count after self-intersection "
                         << "fix at l=" << l << " y=" << y
                         << " (indicates a bug -- please report)\n";
                }
            }
        }

        // ---- accumulate raw swj-unit counts (valid + Type-L + Type-U mixed in) ----
        for (auto& kv : H) {
            ll he = max(0LL, kv.second.first);
            ll ho = max(0LL, kv.second.second);
            Lam_bal += C2(he) + C2(ho);
            Lam_unb += (i128)he * ho;
        }
    }

    // =====================================================================
    // PASS 2: anchor at every u in U -- Type-U acyclic correction.
    // Exact mirror of the Type-L correction above with L and U swapped:
    //   T[t] = (even,odd) wedge counts u -> t  (t in U, via shared l in L)
    //   wedge u-l-t has weight = sign(u,l) XOR sign(l,t)
    // =====================================================================
    for (int u = 0; u < G.nU; u++) {

        unordered_map<int, pair<ll, ll>> T; // t -> (evenWedges, oddWedges)
        T.reserve((size_t)G.degU(u) * 4 + 4);

        for (int l : G.adjU[u]) {
            int s_ul = G.sign(l, u);
            for (int t : G.adjL[l]) {
                if (t == u) continue;
                int s_lt = G.sign(l, t);
                int w = s_ul ^ s_lt;
                auto& p = T[t];
                if (w == 0) p.first++; else p.second++;
            }
        }

        for (auto& kv : T) {
            int t = kv.first;
            ll te = kv.second.first, to = kv.second.second;
            ll ext = (ll)G.degU(t) - 2;
            if (ext > 0) {
                Bar_bal += (C2(te) + C2(to)) * ext;
                Bar_unb += ((i128)te * to) * ext;
            }
        }
    }

    // =====================================================================
    // Final: (Lam - Bar) / 3, per Theorem 2 (each bi-triangle is
    // represented by exactly 3 distinct valid swj-units -- one per
    // antipodal (L,U) vertex pair on the 6-cycle).
    // =====================================================================
    i128 balRaw = Lam_bal - Bar_bal;
    i128 unbRaw = Lam_unb - Bar_unb;

    SwjResult res;
    res.balanced   = balRaw / 3;
    res.unbalanced = unbRaw / 3;

    if (balRaw % 3 != 0 || unbRaw % 3 != 0 ||
        res.balanced < 0 || res.unbalanced < 0) {
        cerr << "WARNING: signedSwjCount internal consistency check failed "
             << "(this indicates a bug -- please report).\n";
        cerr << "  Lam_bal=" << i128_to_string(Lam_bal)
             << " Bar_bal=" << i128_to_string(Bar_bal)
             << " balRaw=" << i128_to_string(balRaw) << "\n";
        cerr << "  Lam_unb=" << i128_to_string(Lam_unb)
             << " Bar_unb=" << i128_to_string(Bar_unb)
             << " unbRaw=" << i128_to_string(unbRaw) << "\n";
    }

    return res;
}

// =========================================================================
// Brute-force ground truth (direct 6-cycle / WJ-style triplet enumeration)
//
// Structurally identical to the common-neighbor-intersection method used
// in the WJ-Count baseline / signed_btc.cpp, adapted to this file's data
// structures and 0/1 sign convention. Only intended for validating the
// SWJ-Count implementation above on small graphs -- cost grows with the
// number of bi-triangles, which can explode combinatorially.
// =========================================================================
struct BruteResult {
    ll total = 0, balanced = 0, unbalanced = 0;
};

static vector<int> commonNeighborsL(const SignedBiGraph& G, int l1, int l2) {
    vector<int> out;
    set_intersection(G.adjL[l1].begin(), G.adjL[l1].end(),
                      G.adjL[l2].begin(), G.adjL[l2].end(),
                      back_inserter(out));
    return out;
}

BruteResult bruteForceCount(const SignedBiGraph& G) {
    BruteResult res;

    for (int l1 = 0; l1 < G.nL; l1++) {
        for (int l2 = l1 + 1; l2 < G.nL; l2++) {

            vector<int> xs = commonNeighborsL(G, l1, l2);
            if (xs.empty()) continue;

            for (int l3 = l2 + 1; l3 < G.nL; l3++) {

                vector<int> ys = commonNeighborsL(G, l2, l3);
                if (ys.empty()) continue;
                vector<int> zs = commonNeighborsL(G, l3, l1);
                if (zs.empty()) continue;

                for (int x : xs) {
                    for (int y : ys) {
                        if (y == x) continue;
                        for (int z : zs) {
                            if (z == x || z == y) continue;

                            // 6-cycle: l1 - x - l2 - y - l3 - z - l1
                            ++res.total;
                            int neg = 0;
                            neg += G.sign(l1, x);
                            neg += G.sign(l2, x);
                            neg += G.sign(l2, y);
                            neg += G.sign(l3, y);
                            neg += G.sign(l3, z);
                            neg += G.sign(l1, z);

                            if (neg % 2 == 0) ++res.balanced;
                            else ++res.unbalanced;
                        }
                    }
                }
            }
        }
    }
    return res;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [--verify]\n";
        return 1;
    }

    SignedBiGraph G;
    if (!loadGraph(argv[1], G)) return 1;

    bool verify = (argc >= 3 && string(argv[2]) == "--verify");

    ll nE = 0;
    for (auto& v : G.adjL) nE += (ll)v.size();
    cout << "Loaded graph: |L|=" << G.nL << "  |U|=" << G.nU
         << "  |E|=" << nE << "\n";

    auto t0 = chrono::high_resolution_clock::now();
    SwjResult res = signedSwjCount(G);
    auto t1 = chrono::high_resolution_clock::now();
    double swjSecs = chrono::duration<double>(t1 - t0).count();

    cout << "\n===== Signed SWJ-Count =====\n";
    cout << "Balanced bi-triangles   : " << i128_to_string(res.balanced) << "\n";
    cout << "Unbalanced bi-triangles : " << i128_to_string(res.unbalanced) << "\n";
    cout << "Total bi-triangles      : " << i128_to_string(res.total()) << "\n";
    cout << "Time                    : " << swjSecs << " sec\n";

    if (verify) {
        cout << "\nRunning brute-force verification (this may be slow)...\n";
        auto b0 = chrono::high_resolution_clock::now();
        BruteResult bf = bruteForceCount(G);
        auto b1 = chrono::high_resolution_clock::now();
        double bfSecs = chrono::duration<double>(b1 - b0).count();

        cout << "\n===== Brute-force (ground truth) =====\n";
        cout << "Balanced bi-triangles   : " << bf.balanced << "\n";
        cout << "Unbalanced bi-triangles : " << bf.unbalanced << "\n";
        cout << "Total bi-triangles      : " << bf.total << "\n";
        cout << "Time                    : " << bfSecs << " sec\n";

        bool ok = (res.balanced == (i128)bf.balanced) &&
                  (res.unbalanced == (i128)bf.unbalanced) &&
                  (res.total() == (i128)bf.total);

        cout << "\n===== Comparison =====\n";
        cout << (ok ? "PASS: signed SWJ-Count matches brute force.\n"
                    : "FAIL: signed SWJ-Count DOES NOT match brute force.\n");
        if (!ok) {
            cout << "  balanced   diff (swj - brute): "
                 << i128_to_string(res.balanced - (i128)bf.balanced) << "\n";
            cout << "  unbalanced diff (swj - brute): "
                 << i128_to_string(res.unbalanced - (i128)bf.unbalanced) << "\n";
            cout << "  total      diff (swj - brute): "
                 << i128_to_string(res.total() - (i128)bf.total) << "\n";
        }
    }

    return 0;
}