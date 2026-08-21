// signed_rswj_count_omp.cpp
//
// OpenMP-parallel SIGNED RSWJ-Count -- counts balanced and unbalanced
// bi-triangles in a signed bipartite graph using the ranked super-wedge
// join method (Yang et al., VLDB 2021, Algorithm 3), extended with
// sign-parity bucketing for the four acyclic-correction lemmas (2-4),
// derived and proved (via edge-disjointness / XOR self-cancellation of
// shared edges) in the accompanying chat derivation.
//
// This is the multi-core version of signed_rswj_count.cpp. The counting
// mathematics is byte-identical to that (validated) file; everything new
// here is scheduling and accumulation:
//
//   * VertexScratch  -- the three per-vertex hash maps (S, T, H) hoisted
//                       out of processVertex*, allocated once per thread
//                       and cleared (not reallocated) per vertex.
//   * WorkItem /     -- L and U merged into ONE fanout-sorted work list,
//     buildWorkList     so the expensive vertices are handed out first
//                       and the dynamic schedule has small work left to
//                       balance with at the tail.
//   * per-thread     -- __int128 has no built-in OpenMP reduction, so
//     accumulators      each thread sums into private i128 locals and
//                       merges once, under a single critical section per
//                       thread (NOT per vertex).
//
// WHY THIS IS SAFE TO PARALLELIZE: RankedCoreGraph is const throughout
// counting and is only ever read (adjacency vectors, rank[], and
// unordered_map::find on signOf -- all const, all thread-safe for
// concurrent readers). Vertices never write to shared state; each one
// only adds into its own thread's accumulators. And because every
// accumulator is an exact integer, addition is associative: the parallel
// result equals the serial result BIT-FOR-BIT regardless of thread count
// or scheduling order. That is what makes --cross-check a meaningful
// test rather than an approximate one.
//
// One cosmetic exception: the "negative H" warnings inside
// processVertex* write to cerr from whichever thread hit them, so if they
// ever fire under parallel execution the lines can interleave. They mean
// the same thing either way, and they should never fire at all -- re-run
// with --serial to read them cleanly.
//
// This file is self-contained: graph loading, (2,2)-core peeling,
// DegRank, rank-filtered neighbor access (Phi/Psi), the signed counting
// logic, and a brute-force validator are all included.
//
// -----------------------------------------------------------------------
// INPUT FORMAT -- same as signed_swj_count.cpp:
//
//   nL nU nE
//   l1 u1 s1
//   ...
//
//   sign: 0 = positive, 1 = negative
// -----------------------------------------------------------------------
//
// BUILD:
//   g++ -O2 -std=c++17 -fopenmp -o signed_rswj_count_omp signed_rswj_count_omp.cpp
//
// The -fopenmp flag is optional: without it the pragmas are ignored and
// the program still builds and runs, single-threaded, giving identical
// results (omp.h is included behind #ifdef _OPENMP for exactly this
// reason). On macOS with Apple Clang, -fopenmp needs libomp:
//   brew install libomp
//   clang++ -O2 -std=c++17 -Xpreprocessor -fopenmp -lomp -o signed_rswj_count_omp signed_rswj_count_omp.cpp
//
// USAGE:
//   ./signed_rswj_count_omp graph.txt                 // parallel (default)
//   ./signed_rswj_count_omp graph.txt --threads=8     // pin thread count
//   ./signed_rswj_count_omp graph.txt --serial        // serial only
//   ./signed_rswj_count_omp graph.txt --cross-check   // serial vs parallel diff
//   ./signed_rswj_count_omp graph.txt --verify        // brute-force diff
//
// --cross-check is the scalable correctness test: brute force only runs
// on tiny graphs, but serial-vs-parallel can be diffed on anything the
// serial version can finish, and the two must agree exactly.
//
// -----------------------------------------------------------------------
// KNOWN ASSUMPTION TO TEST CAREFULLY:
//
// Algorithm 3 line 11 in the source paper reads "foreach z in
// Psi(phi(v),y)", which doesn't parse (phi(v) is a count, not a rank
// threshold). Every other Psi/Phi call in the algorithm uses threshold =
// sigma(v) (the outer anchor's own rank), so this implementation assumes
// it should read Psi(sigma(v), y) -- filtering the H-build's third hop by
// v's rank, consistent with S's build. If brute-force validation fails on
// graphs where a vertex has 3+ hop neighbors with a wide rank spread,
// this is the first place to double check.
// -----------------------------------------------------------------------
//
// DERIVATION SUMMARY (see chat history for full derivation):
//
// A valid rswj-unit (bi-triangle) is balanced iff its two r-super-wedges
// have matching weight (XOR of edge signs) -- same edge-disjointness
// argument as SWJ-Count, just with a unique (not triple) representation
// per bi-triangle (Theorem 3), so NO divide-by-3 here.
//
// Every acyclic correction (Types 1.1, 1.2, 1.3, 2) reduces, via XOR
// self-cancellation of whatever edge/vertex is shared between the two
// halves being compared, to the SAME underlying comparison: the parity
// of the wedge v-x-y (v = anchor, x = first hop, y = second hop). So a
// single (even,odd) split of wedge counts, S_even/S_odd, drives every
// correction term:
//
//   Type-1.1 (Lemma 2) + Type-1.3/2 (Lemma 4), combined (same S-shape,
//   different multiplier, so summed into one multiplier):
//       multiplier = phi(sigma(v),v) + (coreDeg(y) - 2)
//       Bal += (C(S_even,2)+C(S_odd,2)) * multiplier
//       Unb += (S_even*S_odd)          * multiplier
//
//   Type-1.2 (Lemma 3) needs a SECOND, degree-weighted split T_even/T_odd
//   (T[y] += phi(sigma(v),x), routed into the same even/odd bucket as
//   the wedge v-x-y that produced it):
//       Bal += T_even*(S_even-1) + T_odd*(S_odd-1)
//       Unb += T_even*S_odd      + T_odd*S_even
//
//   Self-intersection fix (H-build degenerate case, same pattern as
//   SWJ-Count): weight reduces to just sign(v,z), constant, subtracted
//   psi(sigma(v),z) times from the matching parity bucket.
// -----------------------------------------------------------------------

#include <bits/stdc++.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
typedef long long ll;

// =========================================================================
// 128-bit accumulator support (see signed_swj_count.cpp for full rationale:
// ll tops out ~9.2e18, but the paper's own published bi-triangle counts
// already exceed that on medium real datasets).
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

// ---- packed adjacency entry: neighbour id in the low 31 bits, edge sign
// in the top bit. One sequential load gives both, so the counting loops
// never probe signOf at all.
static inline int PACK(int v, int sgn) { return (int)((unsigned)v | ((unsigned)sgn << 31)); }
static inline int ID(int e)  { return e & 0x7fffffff; }
static inline int SG(int e)  { return (int)((unsigned)e >> 31); }

static inline i128 C2(ll n) { return n > 1 ? (i128)n * (n - 1) / 2 : (i128)0; }

// =========================================================================
// Raw signed bipartite graph (pre-core, as loaded from file)
// =========================================================================
struct RawSignedBiGraph {
    int nL = 0, nU = 0;
    vector<vector<int>> adjL, adjU;
    unordered_map<ll, int> signOf;
    inline ll key(int l, int u) const { return (ll)l * (nU + 1) + u; }
};

bool loadGraph(const char* path, RawSignedBiGraph& G) {
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
// Zero-copy range over a slice of a neighbor list, for Phi/Psi.
// =========================================================================
struct IdxRange {
    const vector<int>* vec = nullptr;
    int lo = 0, hi = 0;

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
// RankedCoreGraph: (2,2)-core + DegRank + signs, all in one place.
// Global id: gidL(l)=l, gidU(u)=nL+u.
// =========================================================================
struct RankedCoreGraph {
    int nL = 0, nU = 0;
    vector<char> aliveL, aliveU;
    vector<vector<int>> adjL, adjU;   // core-filtered; rank-sorted after buildDegRankInPlace
    vector<int> rank;                 // gid -> rank (INT_MAX if dead)
    int nAliveL = 0, nAliveU = 0;
    ll coreEdges = 0;
    unordered_map<ll, int> signOf;    // copied from the raw graph

    inline int gidL(int l) const { return l; }
    inline int gidU(int u) const { return nL + u; }
    inline int rankOfL(int l) const { return rank[gidL(l)]; }
    inline int rankOfU(int u) const { return rank[gidU(u)]; }
    inline int coreDegL(int l) const { return (int)adjL[l].size(); }
    inline int coreDegU(int u) const { return (int)adjU[u].size(); }

    inline ll key(int l, int u) const { return (ll)l * (nU + 1) + u; }
    inline int sign(int l, int u) const {
        auto it = signOf.find(key(l, u));
        assert(it != signOf.end() && "sign() called on a non-existent edge");
        return it->second;
    }

    // Psi(thresholdRank, l): neighbors of l (in U) with rank > thresholdRank
    IdxRange psiL(int l, int thresholdRank) const {
        const auto& lst = adjL[l];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfU(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;
        }
        return IdxRange{&lst, lo, n};
    }
    // Phi(thresholdRank, l): neighbors of l (in U) with rank < thresholdRank
    IdxRange phiL(int l, int thresholdRank) const {
        const auto& lst = adjL[l];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfU(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;
        }
        return IdxRange{&lst, 0, lo};
    }
    // Psi(thresholdRank, u): neighbors of u (in L) with rank > thresholdRank
    IdxRange psiU(int u, int thresholdRank) const {
        const auto& lst = adjU[u];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfL(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;
        }
        return IdxRange{&lst, lo, n};
    }
    // Phi(thresholdRank, u): neighbors of u (in L) with rank < thresholdRank
    IdxRange phiU(int u, int thresholdRank) const {
        const auto& lst = adjU[u];
        int n = (int)lst.size(), lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (rankOfL(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;
        }
        return IdxRange{&lst, 0, lo};
    }
};

// =========================================================================
// (2,2)-core peeling (identical algorithm to Phase A), now also carrying
// signs through into the result.
// =========================================================================
RankedCoreGraph computeCore22(const RawSignedBiGraph& G) {
    RankedCoreGraph R;
    R.nL = G.nL;
    R.nU = G.nU;
    R.aliveL.assign(G.nL, 1);
    R.aliveU.assign(G.nU, 1);
    R.rank.assign(G.nL + G.nU, INT_MAX);
    R.signOf = G.signOf;

    vector<int> degL(G.nL), degU(G.nU);
    for (int l = 0; l < G.nL; l++) degL[l] = (int)G.adjL[l].size();
    for (int u = 0; u < G.nU; u++) degU[u] = (int)G.adjU[u].size();

    queue<pair<int,int>> q; // side: 0=L, 1=U
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
                if (degU[u] == 1) q.push({1, u});
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

    R.adjL.assign(G.nL, {});
    R.adjU.assign(G.nU, {});
    for (int l = 0; l < G.nL; l++) {
        if (!R.aliveL[l]) continue;
        R.nAliveL++;
        for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(PACK(u, R.sign(l, u)));
    }
    for (int u = 0; u < G.nU; u++) {
        if (!R.aliveU[u]) continue;
        R.nAliveU++;
        for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(PACK(l, R.sign(l, u)));
        R.coreEdges += (ll)R.adjU[u].size();
    }

    for (int l = 0; l < G.nL; l++) {
        if (R.aliveL[l] && R.adjL[l].size() < 2) {
            cerr << "WARNING: (2,2)-core invariant violated at L-vertex " << l << "\n";
        }
    }
    for (int u = 0; u < G.nU; u++) {
        if (R.aliveU[u] && R.adjU[u].size() < 2) {
            cerr << "WARNING: (2,2)-core invariant violated at U-vertex " << u << "\n";
        }
    }

    return R;
}

// =========================================================================
// DegRank, built on core degree, plus re-sorting adjacency by neighbor rank.
// =========================================================================
void buildDegRankInPlace(RankedCoreGraph& R) {
    vector<int> gids;
    gids.reserve(R.nAliveL + R.nAliveU);
    for (int l = 0; l < R.nL; l++) if (R.aliveL[l]) gids.push_back(R.gidL(l));
    for (int u = 0; u < R.nU; u++) if (R.aliveU[u]) gids.push_back(R.gidU(u));

    auto coreDegOf = [&](int gid) -> int {
        return (gid < R.nL) ? (int)R.adjL[gid].size() : (int)R.adjU[gid - R.nL].size();
    };

    sort(gids.begin(), gids.end(), [&](int a, int b) {
        int da = coreDegOf(a), db = coreDegOf(b);
        if (da != db) return da > db;
        return a > b;
    });

    for (int i = 0; i < (int)gids.size(); i++) R.rank[gids[i]] = i;

    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        sort(R.adjL[l].begin(), R.adjL[l].end(), [&](int u1, int u2) {
            return R.rankOfU(ID(u1)) < R.rankOfU(ID(u2));
        });
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        sort(R.adjU[u].begin(), R.adjU[u].end(), [&](int l1, int l2) {
            return R.rankOfL(ID(l1)) < R.rankOfL(ID(l2));
        });
    }
}

bool checkRankPermutation(const RankedCoreGraph& R) {
    int nAlive = R.nAliveL + R.nAliveU;
    vector<char> seen(nAlive, 0);
    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        int r = R.rankOfL(l);
        if (r < 0 || r >= nAlive || seen[r]) return false;
        seen[r] = 1;
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        int r = R.rankOfU(u);
        if (r < 0 || r >= nAlive || seen[r]) return false;
        seen[r] = 1;
    }
    return true;
}

// =========================================================================
// Phase B: signed RSWJ-Count
// =========================================================================
struct SwjResult {
    i128 balanced = 0;
    i128 unbalanced = 0;
    i128 total() const { return balanced + unbalanced; }
};

// =========================================================================
// Per-vertex working memory, hoisted out of processVertex* so it can be
// allocated once per thread instead of once per vertex.
//
// These three maps are rebuilt from scratch for every anchor vertex, but
// their bucket arrays are the same size class every time. Declaring them
// locally meant a malloc/free storm -- one full hash table construction
// and destruction per vertex, on every thread, for millions of vertices.
// clear() keeps the buckets and just empties them, so after the first few
// vertices a thread stops allocating almost entirely.
//
// MEMORY TRADEOFF, stated plainly: clear() never shrinks the bucket
// array, so each scratch stays at the high-water mark of the heaviest
// vertex its thread has seen. Because the work list is fanout-descending,
// every thread grabs a heavy vertex first -- so peak memory is reached
// almost immediately and held for the whole run, at T x (largest single
// vertex's S+T+H). That is the intended bargain (bounded by the largest
// neighbourhood, not by the graph), but on a very skewed graph with many
// threads it is the first thing to look at if memory is tight; capping
// --threads is the direct lever.
// =========================================================================
struct VertexScratch {
    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd) wedge counts
    unordered_map<int, pair<ll,ll>> T; // y -> (even,odd) sum of phi(sigma(v),x)
    unordered_map<int, pair<ll,ll>> H; // z -> (even,odd) super-wedge counts
};

// ---- process one L-anchored vertex v=l ---------------------------------
// `scratch` is caller-owned and reused across vertices; it is cleared on
// entry, so callers never need to reset it themselves. It must not be
// shared between threads.
void processVertexL(const RankedCoreGraph& R, int l, VertexScratch& scratch,
                     i128& Lam_bal, i128& Lam_unb, i128& Bar_bal, i128& Bar_unb) {
    int rv = R.rankOfL(l);
    ll phi_v_v = R.phiL(l, rv).size(); // phi(sigma(v),v): v's own neighbors with rank < rv

    // ---- S_even/S_odd (2-hop, y in L) and T_even/T_odd (Lemma 3's weighted sum) ----
    auto& S = scratch.S; // y -> (even,odd) wedge counts
    auto& T = scratch.T; // y -> (even,odd) sum of phi(sigma(v),x)
    S.clear();
    T.clear();

    for (int ex : R.psiL(l, rv)) {               // x in U, rank(x) > rv
        int x = ID(ex), s_lx = SG(ex);
        ll phi_v_x = R.phiU(x, rv).size();        // phi(sigma(v), x)
        for (int ey : R.psiU(x, rv)) {            // y in L, rank(y) > rv  (threshold = rv)
            int y = ID(ey), s_xy = SG(ey);
            int w = s_lx ^ s_xy;                  // wedgeWeight(v,x,y)
            auto& s = S[y];
            auto& t = T[y];
            if (w == 0) { s.first  += 1; t.first  += phi_v_x; }
            else        { s.second += 1; t.second += phi_v_x; }
        }
    }

    // ---- Type-1.1 + Type-1.3/2 (combined) and Type-1.2 corrections ----
    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        ll te = T.at(y).first, to = T.at(y).second;

        ll combinedMult = phi_v_v + ((ll)R.coreDegL(y) - 2);
        if (combinedMult > 0) {
            Bar_bal += (C2(se) + C2(so)) * combinedMult;
            Bar_unb += ((i128)se * so) * combinedMult;
        }

        Bar_bal += (i128)te * (se - 1) + (i128)to * (so - 1);
        Bar_unb += (i128)te * so + (i128)to * se;
    }

    // ---- H_even/H_odd (3-hop, z in U) ----
    auto& H = scratch.H;
    H.clear();
    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        for (int ez : R.psiL(y, rv)) {            // z in U, rank(z) > rv (threshold = rv, see header note)
            int z = ID(ez), s_yz = SG(ez);
            auto& h = H[z];
            if (s_yz == 0) { h.first += se; h.second += so; }
            else           { h.first += so; h.second += se; }
        }
    }

    // ---- self-intersection fix ----
    for (int ez : R.adjL[l]) {                    // z in N(v)
        int z = ID(ez), s_lz = SG(ez);
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiU(z, rv).size();
        if (corr <= 0) continue;
        if (s_lz == 0) {
            it->second.first -= corr;
            if (it->second.first < 0)
                cerr << "WARNING: negative even H at l=" << l << " z=" << z << "\n";
        } else {
            it->second.second -= corr;
            if (it->second.second < 0)
                cerr << "WARNING: negative odd H at l=" << l << " z=" << z << "\n";
        }
    }

    // ---- accumulate Lam ----
    for (auto& kv : H) {
        ll he = max(0LL, kv.second.first);
        ll ho = max(0LL, kv.second.second);
        Lam_bal += C2(he) + C2(ho);
        Lam_unb += (i128)he * ho;
    }
}

// ---- process one U-anchored vertex v=u (exact mirror, L<->U swapped) ---
void processVertexU(const RankedCoreGraph& R, int u, VertexScratch& scratch,
                     i128& Lam_bal, i128& Lam_unb, i128& Bar_bal, i128& Bar_unb) {
    int rv = R.rankOfU(u);
    ll phi_v_v = R.phiU(u, rv).size();

    auto& S = scratch.S; // y -> (even,odd), y in U
    auto& T = scratch.T;
    S.clear();
    T.clear();

    for (int ex : R.psiU(u, rv)) {                // x in L, rank(x) > rv
        int x = ID(ex), s_ux = SG(ex);
        ll phi_v_x = R.phiL(x, rv).size();
        for (int ey : R.psiL(x, rv)) {            // y in U, rank(y) > rv
            int y = ID(ey), s_xy = SG(ey);
            int w = s_ux ^ s_xy;
            auto& s = S[y];
            auto& t = T[y];
            if (w == 0) { s.first  += 1; t.first  += phi_v_x; }
            else        { s.second += 1; t.second += phi_v_x; }
        }
    }

    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        ll te = T.at(y).first, to = T.at(y).second;

        ll combinedMult = phi_v_v + ((ll)R.coreDegU(y) - 2);
        if (combinedMult > 0) {
            Bar_bal += (C2(se) + C2(so)) * combinedMult;
            Bar_unb += ((i128)se * so) * combinedMult;
        }

        Bar_bal += (i128)te * (se - 1) + (i128)to * (so - 1);
        Bar_unb += (i128)te * so + (i128)to * se;
    }

    auto& H = scratch.H;
    H.clear();
    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        for (int ez : R.psiU(y, rv)) {            // z in L, rank(z) > rv
            int z = ID(ez), s_yz = SG(ez);
            auto& h = H[z];
            if (s_yz == 0) { h.first += se; h.second += so; }
            else           { h.first += so; h.second += se; }
        }
    }

    for (int ez : R.adjU[u]) {                    // z in N(v)
        int z = ID(ez), s_uz = SG(ez);
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiL(z, rv).size();
        if (corr <= 0) continue;
        if (s_uz == 0) {
            it->second.first -= corr;
            if (it->second.first < 0)
                cerr << "WARNING: negative even H at u=" << u << " z=" << z << "\n";
        } else {
            it->second.second -= corr;
            if (it->second.second < 0)
                cerr << "WARNING: negative odd H at u=" << u << " z=" << z << "\n";
        }
    }

    for (auto& kv : H) {
        ll he = max(0LL, kv.second.first);
        ll ho = max(0LL, kv.second.second);
        Lam_bal += C2(he) + C2(ho);
        Lam_unb += (i128)he * ho;
    }
}

// ---- shared tail, used by both the serial and the parallel driver ------
static SwjResult finalizeCounts(i128 Lam_bal, i128 Lam_unb,
                                 i128 Bar_bal, i128 Bar_unb,
                                 const char* who) {
    // NOTE: no divide-by-3 here (unlike SWJ-Count) -- Theorem 3 guarantees
    // each bi-triangle has exactly ONE rswj-unit representation, built
    // from its unique minimum-rank vertex.
    SwjResult res;
    res.balanced   = Lam_bal - Bar_bal;
    res.unbalanced = Lam_unb - Bar_unb;

    if (res.balanced < 0 || res.unbalanced < 0) {
        cerr << "WARNING: " << who << " internal consistency check failed "
             << "(this indicates a bug -- please report).\n";
        cerr << "  Lam_bal=" << i128_to_string(Lam_bal) << " Bar_bal=" << i128_to_string(Bar_bal) << "\n";
        cerr << "  Lam_unb=" << i128_to_string(Lam_unb) << " Bar_unb=" << i128_to_string(Bar_unb) << "\n";
    }

    return res;
}

// ---- serial reference implementation -----------------------------------
SwjResult signedRswjCount(const RankedCoreGraph& R) {
    i128 Lam_bal = 0, Lam_unb = 0, Bar_bal = 0, Bar_unb = 0;

    // One scratch for the entire run -- declared once here and reused,
    // never re-declared inside the loop.
    VertexScratch scratch;

    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        processVertexL(R, l, scratch, Lam_bal, Lam_unb, Bar_bal, Bar_unb);
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        processVertexU(R, u, scratch, Lam_bal, Lam_unb, Bar_bal, Bar_unb);
    }

    return finalizeCounts(Lam_bal, Lam_unb, Bar_bal, Bar_unb, "signedRswjCount");
}

// =========================================================================
// Work list: both layers merged, heaviest vertex first.
//
// Per-vertex cost is wildly skewed -- these graphs are power-law, so a
// handful of anchors do more work than the entire tail combined. Two
// consequences shape this list:
//
//   1. L and U go into ONE list rather than two parallel loops. Splitting
//      by layer imposes a barrier between them and lets one layer's tail
//      idle threads that could be starting the other layer's heavy work.
//   2. It is sorted by descending fanout. With a dynamic schedule, the
//      finishing order is governed by whatever is still unclaimed at the
//      end; if a 10-second vertex is handed out last, every other thread
//      waits on it. Heaviest-first means the tail of the run is made of
//      cheap vertices, which is exactly what a dynamic schedule can
//      balance well.
//
// Fanout = own coreDeg + sum of neighbours' coreDeg. This is a proxy, not
// a measurement: real cost is dominated by the S-build, which walks each
// rank-higher neighbour's neighbour list. Degrees are known up front and
// correlate well enough, and getting the ordering roughly right is what
// matters -- the dynamic schedule absorbs the rest.
// =========================================================================
struct WorkItem {
    int side;   // 0 = L-anchored, 1 = U-anchored
    int idx;    // vertex index within its own layer
    ll  fanout; // cost estimate
};

vector<WorkItem> buildWorkList(const RankedCoreGraph& R) {
    vector<WorkItem> work;
    work.reserve((size_t)R.nAliveL + R.nAliveU);

    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        ll f = R.coreDegL(l);
        for (int u : R.adjL[l]) f += R.coreDegU(ID(u));
        work.push_back(WorkItem{0, l, f});
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        ll f = R.coreDegU(u);
        for (int l : R.adjU[u]) f += R.coreDegL(ID(l));
        work.push_back(WorkItem{1, u, f});
    }

    sort(work.begin(), work.end(), [](const WorkItem& a, const WorkItem& b) {
        if (a.fanout != b.fanout) return a.fanout > b.fanout;
        if (a.side != b.side) return a.side < b.side;
        return a.idx < b.idx;   // total order, so the list is reproducible
    });

    return work;
}

// =========================================================================
// Parallel driver.
//
// __int128 is not one of the types OpenMP knows how to reduce, so there is
// no `reduction(+:...)` clause available here. Instead each thread keeps
// private i128 partial sums and merges them once, at the end of its own
// work, inside a critical section -- that is one critical entry per
// thread for the whole run, not one per vertex, so contention is
// irrelevant. Integer addition is exact and associative, so the merge
// order does not affect the result.
// =========================================================================
SwjResult signedRswjCountParallel(const RankedCoreGraph& R,
                                   const vector<WorkItem>& work) {
    i128 Lam_bal = 0, Lam_unb = 0, Bar_bal = 0, Bar_unb = 0;
    const int n = (int)work.size();

#pragma omp parallel
    {
        // Thread-private accumulators and scratch. The scratch in
        // particular MUST be private: it is written on every vertex.
        i128 tLam_bal = 0, tLam_unb = 0, tBar_bal = 0, tBar_unb = 0;
        VertexScratch scratch;

#pragma omp for schedule(dynamic, 4) nowait
        for (int i = 0; i < n; i++) {
            const WorkItem& w = work[i];
            if (w.side == 0)
                processVertexL(R, w.idx, scratch, tLam_bal, tLam_unb, tBar_bal, tBar_unb);
            else
                processVertexU(R, w.idx, scratch, tLam_bal, tLam_unb, tBar_bal, tBar_unb);
        }

#pragma omp critical
        {
            Lam_bal += tLam_bal;
            Lam_unb += tLam_unb;
            Bar_bal += tBar_bal;
            Bar_unb += tBar_unb;
        }
    }

    return finalizeCounts(Lam_bal, Lam_unb, Bar_bal, Bar_unb, "signedRswjCountParallel");
}

// =========================================================================
// Brute-force ground truth, operating on the (2,2)-core (Lemma 5: lossless).
//
// NOTE: adjL/adjU here are sorted by NEIGHBOR RANK (for Phi/Psi), not by
// raw vertex id -- so std::set_intersection's default comparator would
// silently assume the wrong order and could miss matches. Using a
// hash-set intersection instead, which is order-agnostic.
// =========================================================================
struct BruteResult { ll total = 0, balanced = 0, unbalanced = 0; };

static vector<int> commonNeighborsCoreL(const RankedCoreGraph& R, int l1, int l2) {
    unordered_set<int> s1;
    for (int e : R.adjL[l1]) s1.insert(ID(e));
    vector<int> out;
    for (int e : R.adjL[l2]) if (s1.count(ID(e))) out.push_back(ID(e));
    return out;
}

BruteResult bruteForceCount(const RankedCoreGraph& R) {
    BruteResult res;
    for (int l1 = 0; l1 < R.nL; l1++) {
        if (!R.aliveL[l1]) continue;
        for (int l2 = l1 + 1; l2 < R.nL; l2++) {
            if (!R.aliveL[l2]) continue;
            vector<int> xs = commonNeighborsCoreL(R, l1, l2);
            if (xs.empty()) continue;

            for (int l3 = l2 + 1; l3 < R.nL; l3++) {
                if (!R.aliveL[l3]) continue;
                vector<int> ys = commonNeighborsCoreL(R, l2, l3);
                if (ys.empty()) continue;
                vector<int> zs = commonNeighborsCoreL(R, l3, l1);
                if (zs.empty()) continue;

                for (int x : xs) for (int y : ys) {
                    if (y == x) continue;
                    for (int z : zs) {
                        if (z == x || z == y) continue;
                        ++res.total;
                        int neg = 0;
                        neg += R.sign(l1, x);
                        neg += R.sign(l2, x);
                        neg += R.sign(l2, y);
                        neg += R.sign(l3, y);
                        neg += R.sign(l3, z);
                        neg += R.sign(l1, z);
                        if (neg % 2 == 0) ++res.balanced; else ++res.unbalanced;
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
static void printUsage(const char* prog) {
    cerr << "Usage: " << prog << " <graph_file> [options]\n"
         << "\n"
         << "  --threads=N     run the parallel counter with N threads\n"
         << "                  (default: OMP_NUM_THREADS, else all cores)\n"
         << "  --serial        run the serial counter only\n"
         << "  --cross-check   run BOTH counters and diff them; they must\n"
         << "                  agree exactly. Scales to any graph the serial\n"
         << "                  version can finish, unlike --verify.\n"
         << "  --verify        also run brute force (small graphs only)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    bool verify = false, crossCheck = false, forceSerial = false;
    int nThreads = 0;

    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--verify")            verify = true;
        else if (a == "--cross-check")  crossCheck = true;
        else if (a == "--serial")       forceSerial = true;
        else if (a.rfind("--threads=", 0) == 0) {
            nThreads = atoi(a.c_str() + 10);
            if (nThreads < 1) {
                cerr << "ERROR: --threads= requires a positive integer\n";
                return 1;
            }
        } else {
            cerr << "ERROR: unknown option: " << a << "\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (forceSerial && crossCheck) {
        cerr << "ERROR: --serial and --cross-check are mutually exclusive "
             << "(--cross-check already runs the serial counter).\n";
        return 1;
    }

#ifdef _OPENMP
    if (nThreads > 0) omp_set_num_threads(nThreads);
    int maxThreads = omp_get_max_threads();
#else
    int maxThreads = 1;
    if (nThreads > 0) {
        cerr << "NOTE: built WITHOUT OpenMP -- --threads is ignored and the\n"
             << "      \"parallel\" counter runs single-threaded. Rebuild with\n"
             << "      -fopenmp for actual parallelism.\n";
    }
#endif

    RawSignedBiGraph G;
    if (!loadGraph(argv[1], G)) return 1;

    ll origE = 0;
    for (auto& v : G.adjL) origE += (ll)v.size();
    cout << "Loaded graph: |L|=" << G.nL << "  |U|=" << G.nU << "  |E|=" << origE << "\n";
#ifdef _OPENMP
    cout << "OpenMP        : enabled, " << maxThreads << " thread(s)\n";
#else
    cout << "OpenMP        : NOT compiled in (rebuild with -fopenmp)\n";
#endif

    RankedCoreGraph R = computeCore22(G);
    cout << "\n===== (2,2)-core =====\n";
    cout << "Alive L-vertices : " << R.nAliveL << " / " << G.nL << "\n";
    cout << "Alive U-vertices : " << R.nAliveU << " / " << G.nU << "\n";
    cout << "Core edges       : " << R.coreEdges << " / " << origE << "\n";

    buildDegRankInPlace(R);

    if (!checkRankPermutation(R)) {
        cerr << "FATAL: rank[] is not a valid permutation -- aborting before counting.\n";
        return 1;
    }
    cout << "[check] rank[] permutation: PASS\n";

    // ---- work list (only needed for the parallel path) ----
    vector<WorkItem> work;
    if (!forceSerial) {
        auto w0 = chrono::high_resolution_clock::now();
        work = buildWorkList(R);
        auto w1 = chrono::high_resolution_clock::now();

        ll totalFanout = 0;
        for (const WorkItem& w : work) totalFanout += w.fanout;

        cout << "\n===== Work list =====\n";
        cout << "Vertices         : " << work.size() << "\n";
        cout << "Total fanout     : " << totalFanout << "\n";
        if (!work.empty()) {
            cout << "Heaviest vertex  : " << (work[0].side == 0 ? "L" : "U")
                 << " id=" << work[0].idx << "  fanout=" << work[0].fanout
                 << "  (" << fixed << setprecision(2)
                 << (totalFanout > 0 ? 100.0 * (double)work[0].fanout / (double)totalFanout : 0.0)
                 << "% of all work)\n";
            cout.unsetf(ios::fixed);
            cout << setprecision(6);
        }
        cout << "Build time       : "
             << chrono::duration<double>(w1 - w0).count() << " sec\n";
    }

    // ---- run the counter(s) ----
    SwjResult primary;      // the result everything else is compared against
    const char* primaryName = nullptr;
    double parSecs = 0.0, serSecs = 0.0;

    if (!forceSerial) {
        auto t0 = chrono::high_resolution_clock::now();
        primary = signedRswjCountParallel(R, work);
        auto t1 = chrono::high_resolution_clock::now();
        parSecs = chrono::duration<double>(t1 - t0).count();
        primaryName = "parallel";

        cout << "\n===== Signed RSWJ-Count (parallel) =====\n";
        cout << "Balanced bi-triangles   : " << i128_to_string(primary.balanced) << "\n";
        cout << "Unbalanced bi-triangles : " << i128_to_string(primary.unbalanced) << "\n";
        cout << "Total bi-triangles      : " << i128_to_string(primary.total()) << "\n";
        cout << "Threads                 : " << maxThreads << "\n";
        cout << "Time                    : " << parSecs << " sec\n";
    }

    if (forceSerial || crossCheck) {
        auto t0 = chrono::high_resolution_clock::now();
        SwjResult ser = signedRswjCount(R);
        auto t1 = chrono::high_resolution_clock::now();
        serSecs = chrono::duration<double>(t1 - t0).count();

        cout << "\n===== Signed RSWJ-Count (serial) =====\n";
        cout << "Balanced bi-triangles   : " << i128_to_string(ser.balanced) << "\n";
        cout << "Unbalanced bi-triangles : " << i128_to_string(ser.unbalanced) << "\n";
        cout << "Total bi-triangles      : " << i128_to_string(ser.total()) << "\n";
        cout << "Time                    : " << serSecs << " sec\n";

        if (forceSerial) {
            primary = ser;
            primaryName = "serial";
        } else {
            // ---- cross-check: the two must agree BIT-FOR-BIT ----
            bool same = (primary.balanced == ser.balanced) &&
                        (primary.unbalanced == ser.unbalanced);

            cout << "\n===== Cross-check (serial vs parallel) =====\n";
            cout << (same ? "PASS: parallel matches serial exactly.\n"
                          : "FAIL: parallel DOES NOT match serial.\n");
            if (!same) {
                cout << "  balanced   diff (par - ser): "
                     << i128_to_string(primary.balanced - ser.balanced) << "\n";
                cout << "  unbalanced diff (par - ser): "
                     << i128_to_string(primary.unbalanced - ser.unbalanced) << "\n";
                cout << "\nAny difference here is a parallelization bug, not a\n"
                     << "counting-math bug: integer addition is associative, so\n"
                     << "thread count and scheduling cannot change the result.\n"
                     << "Look for shared state -- most likely a VertexScratch\n"
                     << "that escaped the parallel region, or an accumulator\n"
                     << "that ended up shared instead of thread-private.\n";
            }
            if (serSecs > 0.0 && parSecs > 0.0) {
                cout << "  speedup: " << (serSecs / parSecs) << "x on "
                     << maxThreads << " thread(s)\n";
            }
        }
    }

    // ---- brute-force ground truth ----
    if (verify) {
        cout << "\nRunning brute-force verification (this may be slow)...\n";
        auto b0 = chrono::high_resolution_clock::now();
        BruteResult bf = bruteForceCount(R);
        auto b1 = chrono::high_resolution_clock::now();
        double bfSecs = chrono::duration<double>(b1 - b0).count();

        cout << "\n===== Brute-force (ground truth) =====\n";
        cout << "Balanced bi-triangles   : " << bf.balanced << "\n";
        cout << "Unbalanced bi-triangles : " << bf.unbalanced << "\n";
        cout << "Total bi-triangles      : " << bf.total << "\n";
        cout << "Time                    : " << bfSecs << " sec\n";

        bool ok = (primary.balanced == (i128)bf.balanced) &&
                  (primary.unbalanced == (i128)bf.unbalanced) &&
                  (primary.total() == (i128)bf.total);

        cout << "\n===== Comparison (" << primaryName << " vs brute force) =====\n";
        cout << (ok ? "PASS: signed RSWJ-Count matches brute force.\n"
                    : "FAIL: signed RSWJ-Count DOES NOT match brute force.\n");
        if (!ok) {
            cout << "  balanced   diff (rswj - brute): "
                 << i128_to_string(primary.balanced - (i128)bf.balanced) << "\n";
            cout << "  unbalanced diff (rswj - brute): "
                 << i128_to_string(primary.unbalanced - (i128)bf.unbalanced) << "\n";
            cout << "  total      diff (rswj - brute): "
                 << i128_to_string(primary.total() - (i128)bf.total) << "\n";
            cout << "\nHint: if only the split is wrong but the total matches, the\n"
                 << "bug is in sign/parity bucketing (most likely Type-1.2's T-map).\n"
                 << "If the total itself is wrong, the bug is in the structural\n"
                 << "traversal/correction counting -- check the Psi(sigma(v),y)\n"
                 << "assumption noted in the file header first.\n"
                 << "Either way, run --cross-check first: if serial and parallel\n"
                 << "agree, the bug is shared with signed_rswj_count.cpp and is\n"
                 << "NOT in the parallelization.\n";
        }
    }

    return 0;
}
