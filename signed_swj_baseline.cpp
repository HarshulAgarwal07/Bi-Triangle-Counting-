// signed_swj_baseline.cpp
//
// BASELINE: signed SWJ-Count with a brute-force sign extension.
// This is the paper's Algorithm 2 (Yang et al., VLDB 2021, Section 4) extended
// to signed graphs by explicitly checking the sign parity of every candidate
// 6-cycle at the join step. No bucketing, no algebraic acyclic corrections.
//
// Method (per L-anchor l):
//   * Enumerate every 3-hop path (l,u,x,y) with u in U, x in L, y in U.
//     Skip x == l (would reuse edge l-u) and skip y == u (self-intersection).
//     Each path carries its parity (XOR of its 3 edge signs).
//   * Group paths by endpoint y.
//   * For each pair (P1, P2) in a group with P1.u != P2.u and P1.x != P2.x:
//       if P1.par == P2.par -> balanced bi-triangle
//       else                -> unbalanced bi-triangle
//   * Sum over all L-anchors, then divide both counts by 3 (each bi-triangle
//     is seen from its 3 L vertices; divisibility is checked).
//
// This is exactly signed_rswj_baseline.cpp --mode=enum with the rank
// constraint removed. Apples-to-apples with the RSWJ baseline.
//
// PARALLELISM: L-anchors are independent; one vector<Path> per thread.
//
// Cost: O( sum_l sum_y k_{l,y}^2 ) where k_{l,y} = # 3-hop paths from l to y.
// Quadratic in group size, by design.
//
// INPUT FORMAT (same as the other files):
//   nL nU nE
//   l u s          (nE lines; l in [0,nL), u in [0,nU); s: 0 = +, 1 = -)
//
// BUILD:  g++ -O3 -std=c++17 -fopenmp -o signed_swj_baseline signed_swj_baseline.cpp
//         macOS Apple Clang:
//         clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -lomp \
//                 -o signed_swj_baseline signed_swj_baseline.cpp
//
// USAGE:  ./signed_swj_baseline graph.txt [--threads=N] [--core22] [--cross-check] [--verify]
//         --threads=N     thread count (default: OMP_NUM_THREADS, else all cores)
//         --core22        prune to the (2,2)-core first (lossless; handy to match RSWJ)
//         --cross-check   also run with 1 thread and require an exact match
//         --verify        also run brute force (small graphs only)

#include <bits/stdc++.h>
#include <sys/resource.h>
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

static inline int PACK(int v, int s) { return (int)((unsigned)v | ((unsigned)s << 31)); }
static inline int ID(int e)          { return e & 0x7fffffff; }
static inline int SG(int e)          { return (int)((unsigned)e >> 31); }

// ---------------- memory helpers ----------------
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
    if (cur >= 0) cout << fixed << setprecision(1) << setw(10) << toMB(cur) << " MB";
    else          cout << setw(13) << "n/a";
    cout << "   peak RSS: ";
    if (peak >= 0) cout << fixed << setprecision(1) << setw(10) << toMB(peak) << " MB";
    else           cout << setw(13) << "n/a";
    cout << "\n";
    cout.unsetf(ios::fixed); cout << setprecision(6);
}

// ---------------- graph ----------------
struct Graph {
    int nL = 0, nU = 0, n = 0;
    vector<char> alive;
    vector<vector<int>> adj;   // PACK(gid, sign)
    int nAliveL = 0, nAliveU = 0;
    ll coreEdges = 0;
};

static ll graphBytes(const Graph& G) {
    ll b = (ll)G.alive.capacity() + (ll)G.adj.capacity() * (ll)sizeof(vector<int>);
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

static void keepAll(Graph& G) {
    G.alive.assign(G.n, 1);
    G.nAliveL = G.nL; G.nAliveU = G.nU;
    G.coreEdges = 0;
    for (int l = 0; l < G.nL; l++) G.coreEdges += (ll)G.adj[l].size();
}

static void peelCore22(Graph& G) {
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
        a.shrink_to_fit();
        if (g < G.nL) G.nAliveL++; else G.nAliveU++;
        if (g >= G.nL) G.coreEdges += (ll)a.size();
    }
}

// ---------------- counting core ----------------
struct Path { int u, x, y; int par; };

struct Result {
    i128 balanced = 0, unbalanced = 0;
    ll nPaths = 0;
    i128 nPairs = 0;
    ll maxGroup = 0;
    ll maxBufEntries = 0;
    ll maxBufBytes = 0;
    ll sumBufBytes = 0;
    bool divisible = true;
};

static void processAnchor(const Graph& G, int l, vector<Path>& paths, Result& res) {
    paths.clear();
    for (int eu : G.adj[l]) {
        int u = ID(eu), s1 = SG(eu);
        for (int ex : G.adj[u]) {
            int x = ID(ex);
            if (x == l) continue;                     // would reuse edge l-u
            int s2 = s1 ^ SG(ex);
            for (int ey : G.adj[x]) {
                int y = ID(ey);
                if (y == u) continue;                 // self-intersection l-u-x-u
                paths.push_back({u, x, y, s2 ^ SG(ey)});
            }
        }
    }
    res.nPaths += (ll)paths.size();
    if (paths.size() < 2) return;

    sort(paths.begin(), paths.end(),
         [](const Path& a, const Path& b) { return a.y < b.y; });

    i128 bal = 0, unb = 0, pairs = 0;
    ll maxk = 0;
    for (size_t a = 0; a < paths.size();) {
        size_t b = a;
        while (b < paths.size() && paths[b].y == paths[a].y) b++;
        ll k = (ll)(b - a);
        if (k > maxk) maxk = k;
        if (k >= 2) {
            pairs += (i128)k * (k - 1) / 2;
            for (size_t i = a; i < b; i++)
                for (size_t j = i + 1; j < b; j++) {
                    if (paths[i].u == paths[j].u) continue;   // acyclic: shared u
                    if (paths[i].x == paths[j].x) continue;   // acyclic: shared x
                    if (paths[i].par == paths[j].par) bal++;
                    else                              unb++;
                }
        }
        a = b;
    }
    res.balanced   += bal;
    res.unbalanced += unb;
    res.nPairs     += pairs;
    if (maxk > res.maxGroup) res.maxGroup = maxk;
}

struct Work { int gid; ll cost; };

// Only L-anchors: one pass, divisor by 3.
static vector<Work> buildWorkListL(const Graph& G) {
    vector<Work> w;
    w.reserve(G.nL);
    for (int l = 0; l < G.nL; l++) {
        if (!G.alive[l]) continue;
        ll c = 0;
        for (int e : G.adj[l]) c += (ll)G.adj[ID(e)].size();
        w.push_back({l, c});
    }
    sort(w.begin(), w.end(), [](const Work& a, const Work& b) {
        return a.cost != b.cost ? a.cost > b.cost : a.gid < b.gid;
    });
    return w;
}

Result swjCount(const Graph& G, const vector<Work>& work) {
    Result res;
    const int n = (int)work.size();

#pragma omp parallel
    {
        vector<Path> paths;
        Result local;
#pragma omp for schedule(dynamic, 4) nowait
        for (int i = 0; i < n; i++) {
            processAnchor(G, work[i].gid, paths, local);
            local.maxBufEntries = max(local.maxBufEntries, (ll)paths.size());
        }
        ll cap = (ll)paths.capacity() * (ll)sizeof(Path);
#pragma omp critical
        {
            res.balanced     += local.balanced;
            res.unbalanced   += local.unbalanced;
            res.nPaths       += local.nPaths;
            res.nPairs       += local.nPairs;
            res.maxGroup      = max(res.maxGroup,      local.maxGroup);
            res.maxBufEntries = max(res.maxBufEntries, local.maxBufEntries);
            res.maxBufBytes   = max(res.maxBufBytes,   cap);
            res.sumBufBytes  += cap;
        }
    }

    // Each bi-triangle is seen from 3 L-anchors.
    res.divisible = (res.balanced % 3 == 0) && (res.unbalanced % 3 == 0);
    res.balanced   /= 3;
    res.unbalanced /= 3;
    return res;
}

// ---------------- brute force (unchanged) ----------------
struct Brute { ll balanced = 0, unbalanced = 0; };

Brute bruteForce(const Graph& G) {
    vector<unordered_map<int,int>> nb(G.nL);
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

// ---------------- main ----------------
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <graph_file> [--threads=N] [--core22] [--cross-check] [--verify]\n";
        return 1;
    }
    bool verify = false, crossCheck = false, core22 = false;
    int nThreads = 0;
    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--verify") verify = true;
        else if (a == "--cross-check") crossCheck = true;
        else if (a == "--core22") core22 = true;
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
    cout << "Sign handling : brute-force pair check (no bucketing)\n";

    auto p0 = chrono::high_resolution_clock::now();
    if (core22) peelCore22(G); else keepAll(G);
    vector<Work> work = buildWorkListL(G);
    auto p1 = chrono::high_resolution_clock::now();

    cout << "\n===== Input to SWJ (" << (core22 ? "(2,2)-core applied" : "no pruning: faithful SWJ") << ") =====\n";
    cout << "L-vertices : " << G.nAliveL << " / " << G.nL << "\n";
    cout << "U-vertices : " << G.nAliveU << " / " << G.nU << "\n";
    cout << "Edges      : " << G.coreEdges << " / " << origE << "\n";
    cout << "Preprocess time : " << chrono::duration<double>(p1 - p0).count() << " sec\n";
    printMem("after preprocess");

    auto t0 = chrono::high_resolution_clock::now();
    Result r = swjCount(G, work);
    auto t1 = chrono::high_resolution_clock::now();
    double parSecs = chrono::duration<double>(t1 - t0).count();

    cout << "\n===== Signed SWJ-Count (brute-force baseline) =====\n";
    cout << "Balanced bi-triangles   : " << toStr(r.balanced) << "\n";
    cout << "Unbalanced bi-triangles : " << toStr(r.unbalanced) << "\n";
    cout << "Total bi-triangles      : " << toStr(r.balanced + r.unbalanced) << "\n";
    cout << "3-hop paths enumerated  : " << r.nPaths << "\n";
    cout << "Candidate pairs         : " << toStr(r.nPairs) << "  (examined)\n";
    cout << "Largest (l,y) group     : " << r.maxGroup << "\n";
    cout << "[check] balanced and unbalanced divisible by 3 : "
         << (r.divisible ? "PASS" : "FAIL") << "\n";
    cout << "Threads                 : " << maxThreads << "\n";
    cout << "Time                    : " << parSecs << " sec\n";

    ll gB = graphBytes(G);
    ll wB = (ll)work.capacity() * (ll)sizeof(Work);
    cout << "\n===== Memory =====\n";
    cout << fixed << setprecision(1);
    cout << "Graph (containers)            : " << toMB(gB) << " MB\n";
    cout << "Work list                     : " << toMB(wB) << " MB\n";
    cout << "Largest anchor path list      : " << r.maxBufEntries << " entries ("
         << toMB(r.maxBufEntries * (ll)sizeof(Path)) << " MB at " << sizeof(Path) << " B each)\n";
    cout << "Per-thread path buffer, max   : " << toMB(r.maxBufBytes) << " MB\n";
    cout << "Per-thread path buffers, sum  : " << toMB(r.sumBufBytes) << " MB\n";
    cout << "Tracked total (graph + bufs)  : " << toMB(gB + wB + r.sumBufBytes) << " MB\n";
    cout.unsetf(ios::fixed); cout << setprecision(6);
    printMem("after counting");

    if (crossCheck) {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        auto s0 = chrono::high_resolution_clock::now();
        Result ser = swjCount(G, work);
        auto s1 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
        omp_set_num_threads(maxThreads);
#endif
        double serSecs = chrono::duration<double>(s1 - s0).count();
        bool same = ser.balanced == r.balanced && ser.unbalanced == r.unbalanced &&
                    ser.nPaths == r.nPaths && ser.nPairs == r.nPairs;
        cout << "\n===== Cross-check (1 thread vs " << maxThreads << ") =====\n";
        cout << "Serial time             : " << serSecs << " sec\n";
        cout << (same ? "PASS: parallel matches serial exactly.\n"
                      : "FAIL: parallel DOES NOT match serial.\n");
        if (parSecs > 0.0)
            cout << "  speedup: " << (serSecs / parSecs) << "x on " << maxThreads << " thread(s)\n";
        if (!same) return 2;
    }

    if (verify) {
        cout << "\nRunning brute-force verification (this may be slow)...\n";
        Brute b = bruteForce(G);
        bool ok = (r.balanced == (i128)b.balanced) && (r.unbalanced == (i128)b.unbalanced);
        cout << "\n===== Brute-force (ground truth) =====\n";
        cout << "Balanced bi-triangles   : " << b.balanced << "\n";
        cout << "Unbalanced bi-triangles : " << b.unbalanced << "\n";
        cout << (ok ? "PASS: SWJ matches brute force.\n"
                    : "FAIL: SWJ DOES NOT match brute force.\n");
        if (!ok) return 2;
    }
    return r.divisible ? 0 : 3;
}