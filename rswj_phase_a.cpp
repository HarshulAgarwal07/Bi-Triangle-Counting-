// rswj_phase_a.cpp
//
// Phase A of RSWJ-Count (Yang et al., VLDB 2021, Algorithm 3): the
// structural / rank machinery that RSWJ-Count's counting logic (Phase B)
// will run on top of. This file does NOT count anything -- it only builds
// and validates:
//
//   1. The (2,2)-core of the bipartite graph (Lemma 5: every bi-triangle
//      lives inside this core, so pruning to it is lossless).
//   2. DegRank (Definition 6): a single unified priority ranking over
//      BOTH layers, built on CORE degree, where rank 0 = highest degree
//      = highest priority.
//   3. Rank-filtered neighbor access -- Phi(r,v) = neighbors of v with
//      rank < r, Psi(r,v) = neighbors of v with rank > r (Section 5,
//      notation before Definition 8). These are the primitives Phase B's
//      four acyclic-correction lemmas (2-4) and the main r-super-wedge
//      traversal are built from.
//
// -----------------------------------------------------------------------
// INPUT FORMAT -- same as signed_swj_count.cpp, for test-graph reuse:
//
//   nL nU nE
//   l1 u1 s1
//   ...
//
// Phase A ignores the sign field entirely (it's purely structural); it's
// still parsed so existing signed test graphs work unmodified.
// -----------------------------------------------------------------------
//
// USAGE:
//   g++ -O2 -std=c++17 -o rswj_phase_a rswj_phase_a.cpp
//   ./rswj_phase_a graph.txt
//
// Prints core-peeling stats, DegRank diagnostics, and runs self-checks on
// the rank permutation, DegRank's monotonicity property, and Phi/Psi's
// partition property.

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// =========================================================================
// Plain (unsigned, for Phase A purposes) bipartite graph
// =========================================================================
struct BiGraph {
    int nL = 0, nU = 0;
    vector<vector<int>> adjL; // adjL[l] = sorted list of u-indices
    vector<vector<int>> adjU; // adjU[u] = sorted list of l-indices
};

bool loadGraph(const char* path, BiGraph& G) {
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

    for (ll i = 0; i < nE; i++) {
        int l, u, s;
        fin >> l >> u >> s; // sign read and discarded -- Phase A is structural only
        if (!fin) {
            cerr << "ERROR: malformed edge line index " << i << "\n";
            return false;
        }
        if (l < 0 || l >= nL || u < 0 || u >= nU) {
            cerr << "ERROR: edge index out of range at line " << i
                 << " (l=" << l << ", u=" << u << ")\n";
            return false;
        }
        G.adjL[l].push_back(u);
        G.adjU[u].push_back(l);
    }

    for (auto& v : G.adjL) sort(v.begin(), v.end());
    for (auto& v : G.adjU) sort(v.begin(), v.end());

    return true;
}

// =========================================================================
// Lightweight zero-copy range over a slice of a neighbor list, for Phi/Psi.
// =========================================================================
struct IdxRange {
    const vector<int>* vec = nullptr;
    int lo = 0, hi = 0; // half-open [lo, hi) into *vec

    struct It {
        const vector<int>* vec;
        int i;
        int operator*() const { return (*vec)[i]; }
        It& operator++() { ++i; return *this; }
        bool operator!=(const It& o) const { return i != o.i; }
    };
    It begin() const { return It{vec, lo}; }
    It end()   const { return It{vec, hi}; }
    int size() const { return hi - lo; }
};

// =========================================================================
// RankedCoreGraph: the (2,2)-core, with DegRank and rank-sorted adjacency.
//
// Global vertex-id convention (used only internally for the rank[] array):
//   gid(L-vertex l) = l
//   gid(U-vertex u) = nL + u
// =========================================================================
struct RankedCoreGraph {
    int nL = 0, nU = 0;

    vector<char> aliveL, aliveU;      // per ORIGINAL vertex id
    vector<vector<int>> adjL, adjU;   // core-filtered adjacency (only alive
                                       // neighbors); after buildDegRank, also
                                       // sorted by NEIGHBOR's rank ascending

    vector<int> rank;                 // size nL+nU, gid -> rank (INT_MAX if dead)
    int nAliveL = 0, nAliveU = 0;
    ll coreEdges = 0;

    inline int gidL(int l) const { return l; }
    inline int gidU(int u) const { return nL + u; }
    inline int rankOfL(int l) const { return rank[gidL(l)]; }
    inline int rankOfU(int u) const { return rank[gidU(u)]; }
    inline int coreDegL(int l) const { return (int)adjL[l].size(); }
    inline int coreDegU(int u) const { return (int)adjU[u].size(); }

    // ---- Phi/Psi: rank-filtered neighbor access -------------------------
    // adjL[l] / adjU[u] are sorted ascending by neighbor rank once
    // buildDegRank() has run, so these are O(log d) binary search + O(1)
    // per yielded element (no copying).

    // Psi(thresholdRank, l): neighbors of l (in U) with rank > thresholdRank
    IdxRange psiL(int l, int thresholdRank) const {
        const auto& lst = adjL[l];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfU(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;
        }
        return IdxRange{&lst, lo, n};
    }

    // Phi(thresholdRank, l): neighbors of l (in U) with rank < thresholdRank
    IdxRange phiL(int l, int thresholdRank) const {
        const auto& lst = adjL[l];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfU(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;
        }
        return IdxRange{&lst, 0, lo};
    }

    // Psi(thresholdRank, u): neighbors of u (in L) with rank > thresholdRank
    IdxRange psiU(int u, int thresholdRank) const {
        const auto& lst = adjU[u];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfL(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;
        }
        return IdxRange{&lst, lo, n};
    }

    // Phi(thresholdRank, u): neighbors of u (in L) with rank < thresholdRank
    IdxRange phiU(int u, int thresholdRank) const {
        const auto& lst = adjU[u];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfL(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;
        }
        return IdxRange{&lst, 0, lo};
    }
};

// =========================================================================
// Step 1: (2,2)-core peeling.
//
// Standard bucket-queue peeling, O(n+m): repeatedly strip any vertex
// (either layer) with degree < 2, decrementing neighbor degrees, until
// no such vertex remains. Every surviving vertex then has degree >= 2
// within the surviving subgraph -- by Lemma 5 (Yang et al.), every
// bi-triangle is fully contained in this core, so pruning to it loses
// nothing.
// =========================================================================
RankedCoreGraph computeCore22(const BiGraph& G) {
    RankedCoreGraph R;
    R.nL = G.nL;
    R.nU = G.nU;
    R.aliveL.assign(G.nL, 1);
    R.aliveU.assign(G.nU, 1);
    R.rank.assign(G.nL + G.nU, INT_MAX); // filled in later by buildDegRank

    vector<int> degL(G.nL), degU(G.nU);
    for (int l = 0; l < G.nL; l++) degL[l] = (int)G.adjL[l].size();
    for (int u = 0; u < G.nU; u++) degU[u] = (int)G.adjU[u].size();

    // side: 0 = L, 1 = U
    queue<pair<int,int>> q;
    for (int l = 0; l < G.nL; l++) if (degL[l] < 2) q.push({0, l});
    for (int u = 0; u < G.nU; u++) if (degU[u] < 2) q.push({1, u});

    while (!q.empty()) {
        auto [side, idx] = q.front(); q.pop();

        if (side == 0) {
            if (!R.aliveL[idx]) continue;
            R.aliveL[idx] = 0;
            for (int u : G.adjL[idx]) {
                if (!R.aliveU[u]) continue;
                degU[u]--;
                if (degU[u] == 1) q.push({1, u}); // just crossed below threshold
            }
        } else {
            if (!R.aliveU[idx]) continue;
            R.aliveU[idx] = 0;
            for (int l : G.adjU[idx]) {
                if (!R.aliveL[l]) continue;
                degL[l]--;
                if (degL[l] == 1) q.push({0, l});
            }
        }
    }

    // Build core-filtered adjacency (alive vertices only, alive neighbors only).
    R.adjL.assign(G.nL, {});
    R.adjU.assign(G.nU, {});

    for (int l = 0; l < G.nL; l++) {
        if (!R.aliveL[l]) continue;
        R.nAliveL++;
        for (int u : G.adjL[l]) {
            if (R.aliveU[u]) R.adjL[l].push_back(u);
        }
    }
    for (int u = 0; u < G.nU; u++) {
        if (!R.aliveU[u]) continue;
        R.nAliveU++;
        for (int l : G.adjU[u]) {
            if (R.aliveL[l]) R.adjU[u].push_back(l);
        }
        R.coreEdges += (ll)R.adjU[u].size();
    }

    // Self-check: the defining invariant of the (2,2)-core must hold.
    for (int l = 0; l < G.nL; l++) {
        if (R.aliveL[l] && R.adjL[l].size() < 2) {
            cerr << "WARNING: (2,2)-core invariant violated at L-vertex "
                 << l << " (core-degree " << R.adjL[l].size() << ")\n";
        }
    }
    for (int u = 0; u < G.nU; u++) {
        if (R.aliveU[u] && R.adjU[u].size() < 2) {
            cerr << "WARNING: (2,2)-core invariant violated at U-vertex "
                 << u << " (core-degree " << R.adjU[u].size() << ")\n";
        }
    }

    return R;
}

// =========================================================================
// Step 2: DegRank, built on CORE degree (Algorithm 3, line 1: "compute the
// (2,2)-core and build the DegRank on it").
//
// Definition 6: sigma is DegRank if d(u) < d(v) implies sigma(u) > sigma(v).
// I.e. higher (core) degree => smaller rank number => higher priority.
// Ties broken deterministically (direction doesn't matter, just needs to
// be fixed) so the ranking is a genuine total order.
//
// Also re-sorts adjL/adjU by neighbor rank ascending, which is what makes
// psiL/phiL/psiU/phiU above valid O(log d) binary searches.
// =========================================================================
void buildDegRankInPlace(RankedCoreGraph& R) {
    vector<int> gids;
    gids.reserve(R.nAliveL + R.nAliveU);
    for (int l = 0; l < R.nL; l++) if (R.aliveL[l]) gids.push_back(R.gidL(l));
    for (int u = 0; u < R.nU; u++) if (R.aliveU[u]) gids.push_back(R.gidU(u));

    auto coreDegOf = [&](int gid) -> int {
        return (gid < R.nL) ? (int)R.adjL[gid].size()
                             : (int)R.adjU[gid - R.nL].size();
    };

    sort(gids.begin(), gids.end(), [&](int a, int b) {
        int da = coreDegOf(a), db = coreDegOf(b);
        if (da != db) return da > db; // higher core-degree -> earlier -> smaller rank
        return a > b;                 // fixed, arbitrary deterministic tiebreak
    });

    for (int i = 0; i < (int)gids.size(); i++) {
        R.rank[gids[i]] = i;
    }

    // Re-sort core adjacency by neighbor rank ascending.
    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        sort(R.adjL[l].begin(), R.adjL[l].end(), [&](int u1, int u2) {
            return R.rankOfU(u1) < R.rankOfU(u2);
        });
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        sort(R.adjU[u].begin(), R.adjU[u].end(), [&](int l1, int l2) {
            return R.rankOfL(l1) < R.rankOfL(l2);
        });
    }
}

// Convenience: run both steps in one call (what Phase B will actually use).
RankedCoreGraph buildRankedCore(const BiGraph& G) {
    RankedCoreGraph R = computeCore22(G);
    buildDegRankInPlace(R);
    return R;
}

// =========================================================================
// Self-checks (Phase A validation, independent of any Phase B logic)
// =========================================================================
void runSelfChecks(const RankedCoreGraph& R) {
    int nAlive = R.nAliveL + R.nAliveU;
    bool allOk = true;

    // ---- Check 1: rank[] restricted to alive vertices is a permutation
    //      of [0, nAlive) -- no gaps, no duplicates. ----
    {
        vector<char> seen(nAlive, 0);
        bool ok = true;
        for (int l = 0; l < R.nL; l++) {
            if (!R.aliveL[l]) continue;
            int r = R.rankOfL(l);
            if (r < 0 || r >= nAlive || seen[r]) { ok = false; break; }
            seen[r] = 1;
        }
        if (ok) {
            for (int u = 0; u < R.nU; u++) {
                if (!R.aliveU[u]) continue;
                int r = R.rankOfU(u);
                if (r < 0 || r >= nAlive || seen[r]) { ok = false; break; }
                seen[r] = 1;
            }
        }
        cout << "[check] rank[] is a valid permutation of alive vertices: "
             << (ok ? "PASS" : "FAIL") << "\n";
        allOk &= ok;
    }

    // ---- Check 2: DegRank monotonicity (Definition 6) -- spot-check a
    //      bounded number of random alive-vertex pairs (both layers mixed). ----
    {
        vector<int> aliveGids;
        aliveGids.reserve(nAlive);
        for (int l = 0; l < R.nL; l++) if (R.aliveL[l]) aliveGids.push_back(R.gidL(l));
        for (int u = 0; u < R.nU; u++) if (R.aliveU[u]) aliveGids.push_back(R.gidU(u));

        auto coreDegOf = [&](int gid) -> int {
            return (gid < R.nL) ? (int)R.adjL[gid].size()
                                 : (int)R.adjU[gid - R.nL].size();
        };

        mt19937 rng(12345);
        // IMPORTANT: nAlive*(nAlive-1) overflows 32-bit int once nAlive
        // exceeds ~46,340
        ll totalPairsLL = (nAlive > 1) ? (ll)nAlive * (nAlive - 1) / 2 : 0;
        int trials = (int)min<ll>(200000, totalPairsLL);
        bool ok = true;
        for (int t = 0; t < trials && !aliveGids.empty(); t++) {
            int a = aliveGids[rng() % aliveGids.size()];
            int b = aliveGids[rng() % aliveGids.size()];
            if (a == b) continue;
            int da = coreDegOf(a), db = coreDegOf(b);
            int ra = R.rank[a], rb = R.rank[b];
            if (da < db && ra <= rb) { ok = false; break; }
            if (db < da && rb <= ra) { ok = false; break; }
        }
        cout << "[check] DegRank monotonicity (" << trials << " sampled pairs): "
             << (ok ? "PASS" : "FAIL") << "\n";
        allOk &= ok;
    }

    // ---- Check 3: Phi/Psi partition every vertex's neighbor set exactly
    //      (using threshold = the vertex's own rank; every neighbor has a
    //      rank strictly less or strictly greater, never equal, since
    //      ranks are unique). ----
    {
        bool ok = true;
        for (int l = 0; l < R.nL && ok; l++) {
            if (!R.aliveL[l]) continue;
            int r = R.rankOfL(l);
            int total = R.coreDegL(l);
            int phiSz = R.phiL(l, r).size();
            int psiSz = R.psiL(l, r).size();
            if (phiSz + psiSz != total) ok = false;
        }
        for (int u = 0; u < R.nU && ok; u++) {
            if (!R.aliveU[u]) continue;
            int r = R.rankOfU(u);
            int total = R.coreDegU(u);
            int phiSz = R.phiU(u, r).size();
            int psiSz = R.psiU(u, r).size();
            if (phiSz + psiSz != total) ok = false;
        }
        cout << "[check] Phi/Psi partition every neighbor set exactly: "
             << (ok ? "PASS" : "FAIL") << "\n";
        allOk &= ok;
    }

    cout << "[check] Overall Phase A self-check: " << (allOk ? "PASS" : "FAIL") << "\n";
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file>\n";
        return 1;
    }

    BiGraph G;
    if (!loadGraph(argv[1], G)) return 1;

    ll origE = 0;
    for (auto& v : G.adjL) origE += (ll)v.size();
    cout << "Loaded graph: |L|=" << G.nL << "  |U|=" << G.nU
         << "  |E|=" << origE << "\n";

    auto t0 = chrono::high_resolution_clock::now();
    RankedCoreGraph R = computeCore22(G);
    auto t1 = chrono::high_resolution_clock::now();
    double coreSecs = chrono::duration<double>(t1 - t0).count();

    cout << "\n===== (2,2)-core =====\n";
    cout << "Alive L-vertices : " << R.nAliveL << " / " << G.nL << "\n";
    cout << "Alive U-vertices : " << R.nAliveU << " / " << G.nU << "\n";
    cout << "Core edges       : " << R.coreEdges << " / " << origE << "\n";
    cout << "Time             : " << coreSecs << " sec\n";

    auto t2 = chrono::high_resolution_clock::now();
    buildDegRankInPlace(R);
    auto t3 = chrono::high_resolution_clock::now();
    double rankSecs = chrono::duration<double>(t3 - t2).count();

    cout << "\n===== DegRank =====\n";
    cout << "Time             : " << rankSecs << " sec\n";

    // Print the top few highest-priority (rank 0..4) vertices as a sanity sample.
    {
        vector<int> aliveGids;
        for (int l = 0; l < R.nL; l++) if (R.aliveL[l]) aliveGids.push_back(R.gidL(l));
        for (int u = 0; u < R.nU; u++) if (R.aliveU[u]) aliveGids.push_back(R.gidU(u));
        sort(aliveGids.begin(), aliveGids.end(), [&](int a, int b) {
            return R.rank[a] < R.rank[b];
        });

        cout << "Top-priority vertices (side, id, core-degree, rank):\n";
        int shown = 0;
        for (int gid : aliveGids) {
            if (shown++ >= 5) break;
            if (gid < R.nL) {
                cout << "  L  id=" << gid
                     << "  coreDeg=" << R.coreDegL(gid)
                     << "  rank=" << R.rankOfL(gid) << "\n";
            } else {
                int u = gid - R.nL;
                cout << "  U  id=" << u
                     << "  coreDeg=" << R.coreDegU(u)
                     << "  rank=" << R.rankOfU(u) << "\n";
            }
        }
    }

    cout << "\n===== Self-checks =====\n";
    runSelfChecks(R);

    return 0;
}