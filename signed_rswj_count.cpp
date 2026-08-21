// signed_rswj_count.cpp
//
// Phase B: full SIGNED RSWJ-Count -- counts balanced and unbalanced
// bi-triangles in a signed bipartite graph using the ranked super-wedge
// join method (Yang et al., VLDB 2021, Algorithm 3), extended with
// sign-parity bucketing for the four acyclic-correction lemmas (2-4),
// derived and proved (via edge-disjointness / XOR self-cancellation of
// shared edges) in the accompanying chat derivation.
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
// USAGE:
//   g++ -O2 -std=c++17 -o signed_rswj_count signed_rswj_count.cpp
//   ./signed_rswj_count graph.txt              // run RSWJ-Count only
//   ./signed_rswj_count graph.txt --verify      // also brute-force + diff
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
        for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(u);
    }
    for (int u = 0; u < G.nU; u++) {
        if (!R.aliveU[u]) continue;
        R.nAliveU++;
        for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(l);
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

// ---- process one L-anchored vertex v=l ---------------------------------
void processVertexL(const RankedCoreGraph& R, int l,
                     i128& Lam_bal, i128& Lam_unb, i128& Bar_bal, i128& Bar_unb) {
    int rv = R.rankOfL(l);
    ll phi_v_v = R.phiL(l, rv).size(); // phi(sigma(v),v): v's own neighbors with rank < rv

    // ---- S_even/S_odd (2-hop, y in L) and T_even/T_odd (Lemma 3's weighted sum) ----
    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd) wedge counts
    unordered_map<int, pair<ll,ll>> T; // y -> (even,odd) sum of phi(sigma(v),x)

    for (int x : R.psiL(l, rv)) {                // x in U, rank(x) > rv
        int s_lx = R.sign(l, x);
        ll phi_v_x = R.phiU(x, rv).size();        // phi(sigma(v), x)
        for (int y : R.psiU(x, rv)) {             // y in L, rank(y) > rv  (threshold = rv)
            int s_xy = R.sign(y, x);
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
    unordered_map<int, pair<ll,ll>> H;
    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        for (int z : R.psiL(y, rv)) {             // z in U, rank(z) > rv (threshold = rv, see header note)
            int s_yz = R.sign(y, z);
            auto& h = H[z];
            if (s_yz == 0) { h.first += se; h.second += so; }
            else           { h.first += so; h.second += se; }
        }
    }

    // ---- self-intersection fix ----
    for (int z : R.adjL[l]) {                     // z in N(v)
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiU(z, rv).size();
        if (corr <= 0) continue;
        int s_lz = R.sign(l, z);
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
void processVertexU(const RankedCoreGraph& R, int u,
                     i128& Lam_bal, i128& Lam_unb, i128& Bar_bal, i128& Bar_unb) {
    int rv = R.rankOfU(u);
    ll phi_v_v = R.phiU(u, rv).size();

    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd), y in U
    unordered_map<int, pair<ll,ll>> T;

    for (int x : R.psiU(u, rv)) {                 // x in L, rank(x) > rv
        int s_ux = R.sign(x, u);
        ll phi_v_x = R.phiL(x, rv).size();
        for (int y : R.psiL(x, rv)) {             // y in U, rank(y) > rv
            int s_xy = R.sign(x, y);
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

    unordered_map<int, pair<ll,ll>> H;
    for (auto& kv : S) {
        int y = kv.first;
        ll se = kv.second.first, so = kv.second.second;
        for (int z : R.psiU(y, rv)) {             // z in L, rank(z) > rv
            int s_yz = R.sign(z, y);
            auto& h = H[z];
            if (s_yz == 0) { h.first += se; h.second += so; }
            else           { h.first += so; h.second += se; }
        }
    }

    for (int z : R.adjU[u]) {                     // z in N(v)
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiL(z, rv).size();
        if (corr <= 0) continue;
        int s_uz = R.sign(z, u);
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

SwjResult signedRswjCount(const RankedCoreGraph& R) {
    i128 Lam_bal = 0, Lam_unb = 0, Bar_bal = 0, Bar_unb = 0;

    for (int l = 0; l < R.nL; l++) {
        if (!R.aliveL[l]) continue;
        processVertexL(R, l, Lam_bal, Lam_unb, Bar_bal, Bar_unb);
    }
    for (int u = 0; u < R.nU; u++) {
        if (!R.aliveU[u]) continue;
        processVertexU(R, u, Lam_bal, Lam_unb, Bar_bal, Bar_unb);
    }

    // NOTE: no divide-by-3 here (unlike SWJ-Count) -- Theorem 3 guarantees
    // each bi-triangle has exactly ONE rswj-unit representation, built
    // from its unique minimum-rank vertex.
    i128 balRaw = Lam_bal - Bar_bal;
    i128 unbRaw = Lam_unb - Bar_unb;

    SwjResult res;
    res.balanced = balRaw;
    res.unbalanced = unbRaw;

    if (res.balanced < 0 || res.unbalanced < 0) {
        cerr << "WARNING: signedRswjCount internal consistency check failed "
             << "(this indicates a bug -- please report).\n";
        cerr << "  Lam_bal=" << i128_to_string(Lam_bal) << " Bar_bal=" << i128_to_string(Bar_bal) << "\n";
        cerr << "  Lam_unb=" << i128_to_string(Lam_unb) << " Bar_unb=" << i128_to_string(Bar_unb) << "\n";
    }

    return res;
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
    unordered_set<int> s1(R.adjL[l1].begin(), R.adjL[l1].end());
    vector<int> out;
    for (int u : R.adjL[l2]) if (s1.count(u)) out.push_back(u);
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
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [--verify]\n";
        return 1;
    }

    RawSignedBiGraph G;
    if (!loadGraph(argv[1], G)) return 1;

    bool verify = (argc >= 3 && string(argv[2]) == "--verify");

    ll origE = 0;
    for (auto& v : G.adjL) origE += (ll)v.size();
    cout << "Loaded graph: |L|=" << G.nL << "  |U|=" << G.nU << "  |E|=" << origE << "\n";

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

    auto t0 = chrono::high_resolution_clock::now();
    SwjResult res = signedRswjCount(R);
    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    cout << "\n===== Signed RSWJ-Count =====\n";
    cout << "Balanced bi-triangles   : " << i128_to_string(res.balanced) << "\n";
    cout << "Unbalanced bi-triangles : " << i128_to_string(res.unbalanced) << "\n";
    cout << "Total bi-triangles      : " << i128_to_string(res.total()) << "\n";
    cout << "Time                    : " << secs << " sec\n";

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

        bool ok = (res.balanced == (i128)bf.balanced) &&
                  (res.unbalanced == (i128)bf.unbalanced) &&
                  (res.total() == (i128)bf.total);

        cout << "\n===== Comparison =====\n";
        cout << (ok ? "PASS: signed RSWJ-Count matches brute force.\n"
                    : "FAIL: signed RSWJ-Count DOES NOT match brute force.\n");
        if (!ok) {
            cout << "  balanced   diff (rswj - brute): "
                 << i128_to_string(res.balanced - (i128)bf.balanced) << "\n";
            cout << "  unbalanced diff (rswj - brute): "
                 << i128_to_string(res.unbalanced - (i128)bf.unbalanced) << "\n";
            cout << "  total      diff (rswj - brute): "
                 << i128_to_string(res.total() - (i128)bf.total) << "\n";
            cout << "\nHint: if only the split is wrong but the total matches, the\n"
                 << "bug is in sign/parity bucketing (most likely Type-1.2's T-map).\n"
                 << "If the total itself is wrong, the bug is in the structural\n"
                 << "traversal/correction counting -- check the Psi(sigma(v),y)\n"
                 << "assumption noted in the file header first.\n";
        }
    }

    return 0;
}
