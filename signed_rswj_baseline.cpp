// signed_rswj_baseline.cpp
//
// BASELINE signed bi-triangle counter: plain RSWJ, no sign-parity bucketing,
// no closed-form acyclic corrections (Lemmas 2-4). OpenMP-parallel over
// anchor vertices.
//
// What it keeps from RSWJ (Yang et al., VLDB 2021, Section 5):
//   * (2,2)-core pruning                        (Lemma 5)
//   * DegRank vertex ordering                   (same tie-break as the
//                                                bucketed version, so anchors
//                                                are assigned identically)
//   * anchor = minimum-rank vertex of the cycle (Theorem 3: every bi-triangle
//                                                has exactly ONE rswj-unit)
//   * r-super-wedges (v,x,y,z): v is the anchor, x,y,z all have rank > rank(v)
//   * join on the far endpoint z
//
// What it does differently (the whole point of a simple baseline):
//   Instead of counting pairs of r-super-wedges with the closed-form Lemmas
//   2-4, it MATERIALISES every r-super-wedge of the anchor, groups them by z,
//   and resolves the pairs in one of two ways (--mode):
//
//   --mode=enum  (default)  EXPLICIT PAIR CHECK. Look at every pair in a
//       z-group and test  x1!=x2 && y1!=y2  (the pair is a bi-triangle).
//       No parity split anywhere: one parity bit per r-super-wedge, compared
//       pair by pair. Cost ~ #r-super-wedges + #pairs.
//
//   --mode=group  MONARCH-STYLE KEYED AGGREGATION. Monarch (Tang et al.,
//       IEEE BigData 2024) counts butterflies by aggregating wedges under
//       their endpoint-pair key and adding C(count,2) per key instead of
//       enumerating pairs. Same trick here, one level up: the key is the
//       r-super-wedge endpoint z (anchor v is fixed per pass).
//         all pairs   = pairs inside groups keyed (z)
//         acyclic     = pairs inside groups keyed (z,x)   [x1==x2]
//                     + pairs inside groups keyed (z,y)   [y1==y2]
//         bi-triangles = all - acyclic
//       (a pair can't share both x and y -- it would be one super-wedge
//       twice -- so the two acyclic classes are disjoint). Each group needs a
//       parity histogram (c0,c1): balanced = C(c0,2)+C(c1,2), unbalanced =
//       c0*c1. Cost ~ 3 sorts of the r-super-wedge list: no pair loop, no
//       Lemmas 2-4, but sort + materialisation overhead vs. the bucketed code.
//   In both modes: no S/H/T maps, no phi/psi counts, no hand-derived terms.
//
// PARALLELISM: anchors are independent (the graph is read-only while
// counting; each anchor only adds into its thread's accumulators), so the
// anchor loop is a plain `omp parallel for`. Anchors are visited in RANK
// order, which is already heaviest-first (rank 0 = highest degree), so with
// a dynamic schedule the tail of the run is cheap vertices -- no separate
// work list needed. Counts are `long long`, which OpenMP can reduce directly.
// Integer addition is associative, so the result is identical for any thread
// count or schedule. Each thread keeps one reusable r-super-wedge buffer, so
// peak memory is T x (largest single anchor's r-super-wedge list).
//
// Cost (enum): O(#r-super-wedges + sum over (v,z) of k^2) -- it enumerates
// pairs, so it is slower than RSWJ-Count by design.
// Cost (group): O(#r-super-wedges log) -- no quadratic term. Totals can exceed
// 64 bits on big graphs (it counts without enumerating), so accumulators are
// __int128, like the bucketed version.
//
// INPUT FORMAT (same as signed_rswj_count_omp.cpp):
//   nL nU nE
//   l u s          (nE lines; l in [0,nL), u in [0,nU); s: 0 = +, 1 = -)
//
// BUILD:  g++ -O2 -std=c++17 -fopenmp -o signed_rswj_baseline signed_rswj_baseline.cpp
//         (without -fopenmp it still builds and runs single-threaded)
//         macOS Apple Clang: brew install libomp, then
//         clang++ -O2 -std=c++17 -Xpreprocessor -fopenmp -lomp -o signed_rswj_baseline signed_rswj_baseline.cpp
//
// USAGE:  ./signed_rswj_baseline graph.txt [--mode=enum|group] [--threads=N] [--cross-check] [--verify]
//         --mode=...      enum = explicit pair check (default); group = Monarch-style
//         --threads=N     thread count (default: OMP_NUM_THREADS, else all cores)
//         --cross-check   also run with 1 thread and require an exact match
//         --verify        also run brute force (small graphs only)

#include <bits/stdc++.h>
#include <sys/resource.h>   // getrusage (peak RSS)
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
typedef long long ll;
typedef __int128 i128;

static string toStr(i128 v) {
    if (v == 0) return "0";
    bool neg = v < 0; if (neg) v = -v;
    string d;
    while (v > 0) { d += char('0' + (int)(v % 10)); v /= 10; }
    if (neg) d += '-';
    reverse(d.begin(), d.end());
    return d;
}

// Adjacency entry: neighbour gid in the low 31 bits, edge sign in the top bit.
static inline int PACK(int v, int s) { return (int)((unsigned)v | ((unsigned)s << 31)); }
static inline int ID(int e)          { return e & 0x7fffffff; }
static inline int SG(int e)          { return (int)((unsigned)e >> 31); }

// =========================================================================
// Memory profiling helpers.
//   peak RSS    : getrusage(ru_maxrss) -- whole-process high-water mark
//                 (kB on Linux, bytes on macOS).
//   current RSS : /proc/self/status VmRSS -- Linux only (reported as n/a
//                 elsewhere). Freed memory is not always returned to the OS,
//                 so current RSS can stay high after a phase ends; the peak
//                 is the number to trust.
// Peak is process-wide and monotonic: read it right after the phase you care
// about. --cross-check / --verify run AFTER the main count is reported, so
// they can't inflate that line.
// =========================================================================
static double toMB(ll bytes) { return (double)bytes / (1024.0 * 1024.0); }

static ll peakRssBytes() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#ifdef __APPLE__
    return (ll)ru.ru_maxrss;
#else
    return (ll)ru.ru_maxrss * 1024;
#endif
}

static ll currentRssBytes() {
#ifdef __linux__
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line))
        if (line.compare(0, 6, "VmRSS:") == 0) return (ll)atoll(line.c_str() + 6) * 1024;
#endif
    return -1;
}

static void printMem(const char* label) {
    ll cur = currentRssBytes(), peak = peakRssBytes();
    cout << "[mem] " << left << setw(22) << label << right << " current RSS: ";
    if (cur >= 0) cout << fixed << setprecision(1) << setw(10) << toMB(cur) << " MB"; else cout << setw(13) << "n/a";
    cout << "   peak RSS: ";
    if (peak >= 0) cout << fixed << setprecision(1) << setw(10) << toMB(peak) << " MB"; else cout << setw(13) << "n/a";
    cout << "\n";
    cout.unsetf(ios::fixed); cout << setprecision(6);
}

// =========================================================================
// One unified graph over global ids: L vertex l -> l, U vertex u -> nL + u.
// Using one id space means a single counting routine serves both layers
// (the bucketed version needs a processVertexL / processVertexU mirror pair).
// =========================================================================
struct Graph {
    int nL = 0, nU = 0, n = 0;
    vector<char> alive;
    vector<vector<int>> adj;   // PACK(gid, sign); rank-ascending after buildDegRank
    vector<int> rank;          // gid -> rank (INT_MAX if dead)
    vector<int> order;         // alive gids sorted by rank ascending (heaviest first)
    int nAliveL = 0, nAliveU = 0;
    ll coreEdges = 0;
};

// Bytes actually held by the graph's containers (capacity, not size).
static ll graphBytes(const Graph& G) {
    ll b = (ll)G.alive.capacity()
         + (ll)G.rank.capacity()  * (ll)sizeof(int)
         + (ll)G.order.capacity() * (ll)sizeof(int)
         + (ll)G.adj.capacity()   * (ll)sizeof(vector<int>);
    for (const auto& a : G.adj) b += (ll)a.capacity() * (ll)sizeof(int);
    return b;
}

bool loadGraph(const char* path, Graph& G) {
    ifstream fin(path);
    if (!fin) { cerr << "ERROR: cannot open input file: " << path << "\n"; return false; }

    int nL, nU; ll nE;
    fin >> nL >> nU >> nE;
    if (!fin || nL < 0 || nU < 0 || nE < 0) {
        cerr << "ERROR: malformed header (expected: nL nU nE)\n"; return false;
    }
    if ((ll)nL + nU >= (1LL << 31)) { cerr << "ERROR: too many vertices\n"; return false; }

    G.nL = nL; G.nU = nU; G.n = nL + nU;
    G.adj.assign(G.n, {});

    for (ll i = 0; i < nE; i++) {
        int l, u, s;
        fin >> l >> u >> s;
        if (!fin) { cerr << "ERROR: malformed edge line " << i << "\n"; return false; }
        if (l < 0 || l >= nL || u < 0 || u >= nU) {
            cerr << "ERROR: edge index out of range at line " << i << "\n"; return false;
        }
        if (s != 0 && s != 1) { cerr << "ERROR: sign must be 0 or 1 at line " << i << "\n"; return false; }
        G.adj[l].push_back(PACK(nL + u, s));
        G.adj[nL + u].push_back(PACK(l, s));
    }
    return true;
}

// =========================================================================
// (2,2)-core: repeatedly delete vertices of degree < 2, then drop dead
// neighbours from every adjacency list.
// =========================================================================
void peelCore22(Graph& G) {
    vector<int> deg(G.n);
    G.alive.assign(G.n, 1);
    queue<int> q;
    for (int g = 0; g < G.n; g++) {
        deg[g] = (int)G.adj[g].size();
        if (deg[g] < 2) q.push(g);
    }
    while (!q.empty()) {
        int g = q.front(); q.pop();
        if (!G.alive[g]) continue;
        G.alive[g] = 0;
        for (int e : G.adj[g]) {
            int h = ID(e);
            if (G.alive[h] && --deg[h] == 1) q.push(h);
        }
    }
    for (int g = 0; g < G.n; g++) {
        if (!G.alive[g]) { vector<int>().swap(G.adj[g]); continue; }
        auto& a = G.adj[g];
        a.erase(remove_if(a.begin(), a.end(), [&](int e) { return !G.alive[ID(e)]; }), a.end());
        a.shrink_to_fit();   // give back the pruned tail so graphBytes() reflects the core
        if (g < G.nL) G.nAliveL++; else G.nAliveU++;
        if (g >= G.nL) G.coreEdges += (ll)a.size();   // count each edge once (U side)
    }
}

// =========================================================================
// DegRank on core degree (desc), ties by larger gid first -- identical to the
// bucketed version. Then sort each adjacency list by neighbour rank ASCENDING
// so "neighbours with rank > r" is a suffix we can walk from the back.
// =========================================================================
void buildDegRank(Graph& G) {
    vector<int> order;
    for (int g = 0; g < G.n; g++) if (G.alive[g]) order.push_back(g);
    sort(order.begin(), order.end(), [&](int a, int b) {
        size_t da = G.adj[a].size(), db = G.adj[b].size();
        if (da != db) return da > db;
        return a > b;
    });
    G.rank.assign(G.n, INT_MAX);
    for (int i = 0; i < (int)order.size(); i++) G.rank[order[i]] = i;
    G.order = order;
    for (int g : order)
        sort(G.adj[g].begin(), G.adj[g].end(),
             [&](int e1, int e2) { return G.rank[ID(e1)] < G.rank[ID(e2)]; });
}

bool checkRankPermutation(const Graph& G) {
    int nAlive = G.nAliveL + G.nAliveU;
    vector<char> seen(nAlive, 0);
    for (int g = 0; g < G.n; g++) {
        if (!G.alive[g]) continue;
        int r = G.rank[g];
        if (r < 0 || r >= nAlive || seen[r]) return false;
        seen[r] = 1;
    }
    return true;
}

// =========================================================================
// Plain RSWJ with explicit pair checking.
// =========================================================================
enum class Mode { Enum, Group };

struct Result {
    i128 balanced = 0, unbalanced = 0;
    ll nSuperWedges = 0;   // r-super-wedges materialised (paper's "#r-super-wedges")
    i128 nPairs = 0;       // candidate pairs (incl. acyclic) -- examined in enum, counted in group
    // ---- memory stats for the r-super-wedge buffers ----
    ll maxAnchorSW = 0;    // most r-super-wedges held for a single anchor (entries)
    ll maxBufBytes = 0;    // largest per-thread buffer capacity (bytes)
    ll sumBufBytes = 0;    // sum of per-thread buffer capacities (bytes)
    i128 total() const { return balanced + unbalanced; }
};

struct SW { int z, x, y; int par; };   // r-super-wedge (v,x,y,z) of the current anchor v

// Sort `sws` by `less`; for every maximal run of equal key (`same`) add the
// pairs INSIDE the run, split by parity. This is Monarch's "aggregate by key,
// then C(count,2)" with a parity histogram per key.
//   same parity -> balanced:    C(c0,2) + C(c1,2)
//   diff parity -> unbalanced:  c0 * c1
template <class Less, class Same>
static void keyedPairs(vector<SW>& sws, Less less, Same same, i128& bal, i128& unb) {
    sort(sws.begin(), sws.end(), less);
    for (size_t a = 0; a < sws.size();) {
        size_t b = a;
        ll c[2] = {0, 0};
        while (b < sws.size() && same(sws[a], sws[b])) { c[sws[b].par]++; b++; }
        bal += (i128)c[0] * (c[0] - 1) / 2 + (i128)c[1] * (c[1] - 1) / 2;
        unb += (i128)c[0] * c[1];
        a = b;
    }
}

// Process ONE anchor v. Reads G only; writes only to the caller's buffer and
// counters, so it is safe to run for different anchors on different threads.
static void processAnchor(const Graph& G, int v, Mode mode, vector<SW>& sws,
                          i128& bal, i128& unb, i128& pairs, ll& nSW) {
    const int rv = G.rank[v];
    sws.clear();

    // ---- enumerate r-super-wedges (v,x,y,z): x,y,z all rank > rv ----
    // adj lists are rank-ascending, so walk from the back and stop at the
    // first neighbour with rank <= rv.
    const auto& av = G.adj[v];
    for (int i = (int)av.size() - 1; i >= 0; --i) {
        int x = ID(av[i]);
        if (G.rank[x] <= rv) break;
        int p1 = SG(av[i]);

        const auto& ax = G.adj[x];
        for (int j = (int)ax.size() - 1; j >= 0; --j) {
            int y = ID(ax[j]);
            if (G.rank[y] <= rv) break;          // also excludes y == v
            int p2 = p1 ^ SG(ax[j]);

            const auto& ay = G.adj[y];
            for (int k = (int)ay.size() - 1; k >= 0; --k) {
                int z = ID(ay[k]);
                if (G.rank[z] <= rv) break;
                if (z == x) continue;            // v-x-y-x is not a simple path
                sws.push_back(SW{z, x, y, p2 ^ SG(ay[k])});
            }
        }
    }
    nSW += (ll)sws.size();
    if (sws.size() < 2) return;

    if (mode == Mode::Group) {
        // ---- Monarch-style: aggregate by key, C(count,2) per key ----
        i128 tb = 0, tu = 0;   // all pairs sharing (v,z)
        i128 xb = 0, xu = 0;   // pairs sharing (v,z) AND x   (acyclic)
        i128 yb = 0, yu = 0;   // pairs sharing (v,z) AND y   (acyclic)
        keyedPairs(sws,
            [](const SW& a, const SW& b) { return a.z < b.z; },
            [](const SW& a, const SW& b) { return a.z == b.z; }, tb, tu);
        keyedPairs(sws,
            [](const SW& a, const SW& b) { return a.z != b.z ? a.z < b.z : a.x < b.x; },
            [](const SW& a, const SW& b) { return a.z == b.z && a.x == b.x; }, xb, xu);
        keyedPairs(sws,
            [](const SW& a, const SW& b) { return a.z != b.z ? a.z < b.z : a.y < b.y; },
            [](const SW& a, const SW& b) { return a.z == b.z && a.y == b.y; }, yb, yu);
        bal   += tb - xb - yb;
        unb   += tu - xu - yu;
        pairs += tb + tu;
        return;
    }

    // ---- enum: explicit pair check inside each z-group ----
    sort(sws.begin(), sws.end(), [](const SW& a, const SW& b) { return a.z < b.z; });
    for (size_t a = 0; a < sws.size();) {
        size_t b = a;
        while (b < sws.size() && sws[b].z == sws[a].z) b++;
        pairs += (i128)(b - a) * (i128)(b - a - 1) / 2;

        for (size_t i = a; i < b; i++)
            for (size_t j = i + 1; j < b; j++) {
                // cyclic iff the two halves share no middle vertex
                if (sws[i].x == sws[j].x || sws[i].y == sws[j].y) continue;
                if (sws[i].par == sws[j].par) bal++;
                else                          unb++;
            }
        a = b;
    }
}

Result rswjBaseline(const Graph& G, Mode mode) {
    Result res;
    const int n = (int)G.order.size();

#pragma omp parallel
    {
        vector<SW> sws;   // thread-private, reused across this thread's anchors
        i128 bal = 0, unb = 0, pairs = 0;
        ll nSW = 0, maxSW = 0;

#pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < n; i++) {
            processAnchor(G, G.order[i], mode, sws, bal, unb, pairs, nSW);
            maxSW = max(maxSW, (ll)sws.size());   // sws still holds this anchor's list
        }

        ll cap = (ll)sws.capacity() * (ll)sizeof(SW);   // this thread's high-water mark
#pragma omp critical
        {
            res.balanced += bal; res.unbalanced += unb; res.nPairs += pairs;
            res.nSuperWedges += nSW;
            res.maxAnchorSW = max(res.maxAnchorSW, maxSW);
            res.maxBufBytes = max(res.maxBufBytes, cap);
            res.sumBufBytes += cap;
        }
    }
    return res;
}

// =========================================================================
// Brute-force ground truth on the (2,2)-core (same method as the bucketed
// file's validator): for each L-triple l1<l2<l3, choose distinct common
// neighbours x in N(l1)&N(l2), y in N(l2)&N(l3), z in N(l3)&N(l1).
// =========================================================================
struct Brute { ll balanced = 0, unbalanced = 0; };

Brute bruteForce(const Graph& G) {
    vector<unordered_map<int,int>> nb(G.nL);          // l -> (u gid -> sign)
    for (int l = 0; l < G.nL; l++)
        if (G.alive[l]) for (int e : G.adj[l]) nb[l][ID(e)] = SG(e);

    struct C { int w, par; };
    auto common = [&](int a, int b) {
        vector<C> out;
        for (int e : G.adj[a]) {
            auto it = nb[b].find(ID(e));
            if (it != nb[b].end()) out.push_back(C{ID(e), SG(e) ^ it->second});
        }
        return out;
    };

    Brute res;
    for (int l1 = 0; l1 < G.nL; l1++) {
        if (!G.alive[l1]) continue;
        for (int l2 = l1 + 1; l2 < G.nL; l2++) {
            if (!G.alive[l2]) continue;
            auto xs = common(l1, l2);
            if (xs.empty()) continue;
            for (int l3 = l2 + 1; l3 < G.nL; l3++) {
                if (!G.alive[l3]) continue;
                auto ys = common(l2, l3); if (ys.empty()) continue;
                auto zs = common(l3, l1); if (zs.empty()) continue;
                for (auto& x : xs) for (auto& y : ys) {
                    if (y.w == x.w) continue;
                    for (auto& z : zs) {
                        if (z.w == x.w || z.w == y.w) continue;
                        if ((x.par ^ y.par ^ z.par) == 0) res.balanced++;
                        else                              res.unbalanced++;
                    }
                }
            }
        }
    }
    return res;
}

// =========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [--mode=enum|group] [--threads=N] [--cross-check] [--verify]\n";
        return 1;
    }
    bool verify = false, crossCheck = false;
    int nThreads = 0;
    Mode mode = Mode::Enum;
    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--verify") verify = true;
        else if (a == "--mode=enum")  mode = Mode::Enum;
        else if (a == "--mode=group") mode = Mode::Group;
        else if (a == "--cross-check") crossCheck = true;
        else if (a.rfind("--threads=", 0) == 0) {
            nThreads = atoi(a.c_str() + 10);
            if (nThreads < 1) { cerr << "ERROR: --threads= requires a positive integer\n"; return 1; }
        }
        else { cerr << "ERROR: unknown option: " << a << "\n"; return 1; }
    }

#ifdef _OPENMP
    if (nThreads > 0) omp_set_num_threads(nThreads);
    const int maxThreads = omp_get_max_threads();
#else
    const int maxThreads = 1;
    if (nThreads > 0)
        cerr << "NOTE: built WITHOUT OpenMP -- --threads is ignored (rebuild with -fopenmp).\n";
#endif

    Graph G;
    if (!loadGraph(argv[1], G)) return 1;
    ll origE = 0;
    for (int l = 0; l < G.nL; l++) origE += (ll)G.adj[l].size();
    printMem("after load");
    cout << "Loaded graph: |L|=" << G.nL << "  |U|=" << G.nU << "  |E|=" << origE << "\n";
#ifdef _OPENMP
    cout << "OpenMP        : enabled, " << maxThreads << " thread(s)\n";
#else
    cout << "OpenMP        : NOT compiled in (rebuild with -fopenmp)\n";
#endif

    auto p0 = chrono::high_resolution_clock::now();
    peelCore22(G);
    buildDegRank(G);
    auto p1 = chrono::high_resolution_clock::now();

    cout << "\n===== (2,2)-core =====\n";
    cout << "Alive L-vertices : " << G.nAliveL << " / " << G.nL << "\n";
    cout << "Alive U-vertices : " << G.nAliveU << " / " << G.nU << "\n";
    cout << "Core edges       : " << G.coreEdges << " / " << origE << "\n";
    cout << "Preprocess time  : " << chrono::duration<double>(p1 - p0).count() << " sec\n";

    if (!checkRankPermutation(G)) { cerr << "FATAL: rank[] is not a valid permutation\n"; return 1; }
    cout << "[check] rank[] permutation: PASS\n";
    printMem("after core + rank");

    auto t0 = chrono::high_resolution_clock::now();
    Result r = rswjBaseline(G, mode);
    auto t1 = chrono::high_resolution_clock::now();

    double parSecs = chrono::duration<double>(t1 - t0).count();

    cout << "\n===== Signed RSWJ baseline, mode="
         << (mode == Mode::Group ? "group (Monarch-style keyed aggregation)" : "enum (explicit pair check)")
         << " =====\n";
    cout << "Balanced bi-triangles   : " << toStr(r.balanced) << "\n";
    cout << "Unbalanced bi-triangles : " << toStr(r.unbalanced) << "\n";
    cout << "Total bi-triangles      : " << toStr(r.total()) << "\n";
    cout << "r-super-wedges          : " << r.nSuperWedges << "\n";
    cout << "Candidate pairs         : " << toStr(r.nPairs)
         << (mode == Mode::Group ? "  (counted, not examined)" : "  (examined)") << "\n";
    if (r.nSuperWedges > 0)
        cout << "Pairs / r-super-wedge   : " << fixed << setprecision(2)
             << (double)r.nPairs / (double)r.nSuperWedges << "\n";
    cout.unsetf(ios::fixed); cout << setprecision(6);
    cout << "Threads                 : " << maxThreads << "\n";
    cout << "Time                    : " << parSecs << " sec\n";

    // ---- memory report (read BEFORE cross-check / brute force run) ----
    ll gB = graphBytes(G);
    cout << "\n===== Memory =====\n";
    cout << fixed << setprecision(1);
    cout << "Core graph (containers)       : " << toMB(gB) << " MB\n";
    cout << "Largest anchor SW list        : " << r.maxAnchorSW << " entries ("
         << toMB(r.maxAnchorSW * (ll)sizeof(SW)) << " MB at " << sizeof(SW) << " B each)\n";
    cout << "Per-thread SW buffer, max     : " << toMB(r.maxBufBytes) << " MB\n";
    cout << "Per-thread SW buffers, sum    : " << toMB(r.sumBufBytes) << " MB\n";
    cout << "Tracked total (graph + bufs)  : " << toMB(gB + r.sumBufBytes) << " MB\n";
    cout.unsetf(ios::fixed); cout << setprecision(6);
    printMem("after counting");

    if (crossCheck) {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        auto s0 = chrono::high_resolution_clock::now();
        Result ser = rswjBaseline(G, mode);
        auto s1 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
        omp_set_num_threads(maxThreads);
#endif
        double serSecs = chrono::duration<double>(s1 - s0).count();
        bool same = (ser.balanced == r.balanced) && (ser.unbalanced == r.unbalanced) &&
                    (ser.nSuperWedges == r.nSuperWedges) && (ser.nPairs == r.nPairs);
        cout << "\n===== Cross-check (1 thread vs " << maxThreads << ") =====\n";
        cout << "Serial time             : " << serSecs << " sec\n";
        cout << (same ? "PASS: parallel matches serial exactly.\n"
                      : "FAIL: parallel DOES NOT match serial.\n");
        if (parSecs > 0.0) cout << "  speedup: " << (serSecs / parSecs) << "x on " << maxThreads << " thread(s)\n";
        if (!same) return 2;
    }

    if (verify) {
        cout << "\nRunning brute-force verification (this may be slow)...\n";
        Brute b = bruteForce(G);
        bool ok = (r.balanced == (i128)b.balanced) && (r.unbalanced == (i128)b.unbalanced);
        cout << "\n===== Brute-force (ground truth) =====\n";
        cout << "Balanced bi-triangles   : " << b.balanced << "\n";
        cout << "Unbalanced bi-triangles : " << b.unbalanced << "\n";
        cout << (ok ? "PASS: baseline matches brute force.\n"
                    : "FAIL: baseline DOES NOT match brute force.\n");
        if (!ok) return 2;
    }
    return 0;
}
