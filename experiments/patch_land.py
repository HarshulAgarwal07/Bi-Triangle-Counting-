# patch_land.py -- applies BOTH landed optimizations to a main source file.
#
#   1. pack the edge sign into the top bit of each adjacency entry, so the
#      counting loops never probe signOf (was ~74% of serial runtime);
#   2. merge the S and T maps, which are always keyed identically yet were
#      probed three times per y, into one map with a 4-field value.
#
# This supersedes patch_packed.py + patch_st.py, which only handled the OMP
# file. It handles BOTH signed_rswj_count.cpp (S/T/H are function locals) and
# signed_rswj_count_omp.cpp (S/T/H live in VertexScratch), so the two files
# provably get identical treatment in the counting path.
#
# Kept after landing so the transformation is re-runnable and auditable, not
# because the sources still need it. Every substitution asserts its expected
# hit count: if a source changes shape, this fails loudly rather than
# silently producing something subtly wrong.
#
#   python3 patch_land.py ../signed_rswj_count.cpp     /tmp/out.cpp
#   python3 patch_land.py ../signed_rswj_count_omp.cpp /tmp/out.cpp

import io, sys

src = sys.argv[1]
dst = sys.argv[2]

raw = io.open(src, encoding='utf-8', newline='').read()
crlf = '\r\n' in raw
s = raw.replace('\r\n', '\n')


def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "expected %d got %d for: %r" % (count, n, old[:80])
    s = s.replace(old, new)


def sub_opt(old, new, count=1):
    """For text present in only one of the two files (plumbing differences)."""
    global s
    n = s.count(old)
    assert n in (0, count), "expected 0 or %d got %d for: %r" % (count, n, old[:80])
    if n:
        s = s.replace(old, new)
    return n > 0


# =====================================================================
# 1. packed adjacency entry
# =====================================================================

sub('static inline i128 C2(ll n)',
    '''// ---- packed adjacency entry: neighbour id in the low 31 bits, edge sign
// in the top bit. One sequential load gives both, so the counting loops
// never probe signOf at all.
//
// This is sound because every sign() call in the counting path asks for the
// sign of an edge that is ALREADY BEING TRAVERSED -- the sign of the very
// adjacency entry just read -- so it can be carried in the entry itself
// instead of being looked up again by (l,u) key. signOf survives on the
// struct for the brute-force validator, but the hot path never touches it.
//
// Ids are < 2^31 (vertex counts are ints), so bit 31 is free.
static inline int PACK(int v, int sgn) { return (int)((unsigned)v | ((unsigned)sgn << 31)); }
static inline int ID(int e)  { return e & 0x7fffffff; }
static inline int SG(int e)  { return (int)((unsigned)e >> 31); }

static inline i128 C2(ll n)''')

# ---- Phi/Psi binary searches: compare the rank of the UNPACKED id ----
sub('if (rankOfU(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;',
    'if (rankOfU(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;')
sub('if (rankOfU(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;',
    'if (rankOfU(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;')
sub('if (rankOfL(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;',
    'if (rankOfL(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;')
sub('if (rankOfL(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;',
    'if (rankOfL(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;')

# ---- core build: attach the sign while filtering to the (2,2)-core ----
sub('for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(u);',
    'for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(PACK(u, R.sign(l, u)));')
sub('for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(l);',
    'for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(PACK(l, R.sign(l, u)));')

# ---- rank sort compares unpacked ids ----
sub('return R.rankOfU(u1) < R.rankOfU(u2);', 'return R.rankOfU(ID(u1)) < R.rankOfU(ID(u2));')
sub('return R.rankOfL(l1) < R.rankOfL(l2);', 'return R.rankOfL(ID(l1)) < R.rankOfL(ID(l2));')

# =====================================================================
# 2. merged S/T value type
# =====================================================================

WEDGE = '''struct Wedge {
    ll se = 0, so = 0;   // wedge counts, by parity        (was the S map)
    ll te = 0, to = 0;   // sum of phi(sigma(v),x), by parity (was the T map)
};

'''

# OMP file: S/T/H live in VertexScratch. Serial file: they are locals.
is_omp = sub_opt('''struct VertexScratch {
    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd) wedge counts
    unordered_map<int, pair<ll,ll>> T; // y -> (even,odd) sum of phi(sigma(v),x)
    unordered_map<int, pair<ll,ll>> H; // z -> (even,odd) super-wedge counts
};''',
    WEDGE + '''struct VertexScratch {
    unordered_map<int, Wedge> S;       // y -> wedge counts AND phi-weighted sums
    unordered_map<int, pair<ll,ll>> H; // z -> (even,odd) super-wedge counts
};''')

if is_omp:
    sub('''    auto& S = scratch.S; // y -> (even,odd) wedge counts
    auto& T = scratch.T; // y -> (even,odd) sum of phi(sigma(v),x)
    S.clear();
    T.clear();''',
        '''    auto& S = scratch.S; // y -> wedge counts AND phi-weighted sums
    S.clear();''')
    sub('''    auto& S = scratch.S; // y -> (even,odd), y in U
    auto& T = scratch.T;
    S.clear();
    T.clear();''',
        '''    auto& S = scratch.S; // y -> wedge counts AND phi-weighted sums, y in U
    S.clear();''')
else:
    sub('// ---- process one L-anchored vertex v=l ---------------------------------',
        WEDGE + '// ---- process one L-anchored vertex v=l ---------------------------------')
    sub('''    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd) wedge counts
    unordered_map<int, pair<ll,ll>> T; // y -> (even,odd) sum of phi(sigma(v),x)''',
        '    unordered_map<int, Wedge> S;       // y -> wedge counts AND phi-weighted sums')
    sub('''    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd), y in U
    unordered_map<int, pair<ll,ll>> T;''',
        '    unordered_map<int, Wedge> S;       // y -> wedge counts AND phi-weighted sums, y in U')

# =====================================================================
# 3. counting bodies -- byte-identical text in both files
# =====================================================================

# ---- processVertexL: S/T build ----
sub('''    for (int x : R.psiL(l, rv)) {                // x in U, rank(x) > rv
        int s_lx = R.sign(l, x);
        ll phi_v_x = R.phiU(x, rv).size();        // phi(sigma(v), x)
        for (int y : R.psiU(x, rv)) {             // y in L, rank(y) > rv  (threshold = rv)
            int s_xy = R.sign(y, x);''',
    '''    for (int ex : R.psiL(l, rv)) {               // x in U, rank(x) > rv
        int x = ID(ex), s_lx = SG(ex);
        ll phi_v_x = R.phiU(x, rv).size();        // phi(sigma(v), x)
        for (int ey : R.psiU(x, rv)) {            // y in L, rank(y) > rv  (threshold = rv)
            int y = ID(ey), s_xy = SG(ey);''')

# ---- processVertexU: S/T build ----
sub('''    for (int x : R.psiU(u, rv)) {                 // x in L, rank(x) > rv
        int s_ux = R.sign(x, u);
        ll phi_v_x = R.phiL(x, rv).size();
        for (int y : R.psiL(x, rv)) {             // y in U, rank(y) > rv
            int s_xy = R.sign(x, y);''',
    '''    for (int ex : R.psiU(u, rv)) {                // x in L, rank(x) > rv
        int x = ID(ex), s_ux = SG(ex);
        ll phi_v_x = R.phiL(x, rv).size();
        for (int ey : R.psiL(x, rv)) {            // y in U, rank(y) > rv
            int y = ID(ey), s_xy = SG(ey);''')

# ---- one probe instead of two, in both ----
sub('''            auto& s = S[y];
            auto& t = T[y];
            if (w == 0) { s.first  += 1; t.first  += phi_v_x; }
            else        { s.second += 1; t.second += phi_v_x; }''',
    '''            auto& s = S[y];
            if (w == 0) { s.se += 1; s.te += phi_v_x; }
            else        { s.so += 1; s.to += phi_v_x; }''', 2)

# ---- correction loop: drops the third probe (T.at(y)) entirely ----
sub('''        ll se = kv.second.first, so = kv.second.second;
        ll te = T.at(y).first, to = T.at(y).second;''',
    '''        ll se = kv.second.se, so = kv.second.so;
        ll te = kv.second.te, to = kv.second.to;''', 2)

# ---- H build: read the new field names, and unpack z/s_yz from the entry ----
sub('''        ll se = kv.second.first, so = kv.second.second;
        for (int z : R.psiL(y, rv)) {             // z in U, rank(z) > rv (threshold = rv, see header note)
            int s_yz = R.sign(y, z);''',
    '''        ll se = kv.second.se, so = kv.second.so;
        for (int ez : R.psiL(y, rv)) {            // z in U, rank(z) > rv (threshold = rv, see header note)
            int z = ID(ez), s_yz = SG(ez);''')
sub('''        ll se = kv.second.first, so = kv.second.second;
        for (int z : R.psiU(y, rv)) {             // z in L, rank(z) > rv
            int s_yz = R.sign(z, y);''',
    '''        ll se = kv.second.se, so = kv.second.so;
        for (int ez : R.psiU(y, rv)) {            // z in L, rank(z) > rv
            int z = ID(ez), s_yz = SG(ez);''')

# ---- self-intersection fix: sign comes from the entry, not a lookup ----
sub('''    for (int z : R.adjL[l]) {                     // z in N(v)
        auto it = H.find(z);''',
    '''    for (int ez : R.adjL[l]) {                    // z in N(v)
        int z = ID(ez), s_lz = SG(ez);
        auto it = H.find(z);''')
sub('        int s_lz = R.sign(l, z);\n', '')
sub('''    for (int z : R.adjU[u]) {                     // z in N(v)
        auto it = H.find(z);''',
    '''    for (int ez : R.adjU[u]) {                    // z in N(v)
        int z = ID(ez), s_uz = SG(ez);
        auto it = H.find(z);''')
sub('        int s_uz = R.sign(z, u);\n', '')

# =====================================================================
# 4. remaining adjacency consumers outside the counting path
# =====================================================================

# work list fanout (OMP only)
sub_opt('for (int u : R.adjL[l]) f += R.coreDegU(u);',
        'for (int u : R.adjL[l]) f += R.coreDegU(ID(u));')
sub_opt('for (int l : R.adjU[u]) f += R.coreDegL(l);',
        'for (int l : R.adjU[u]) f += R.coreDegL(ID(l));')

# brute-force validator (both files)
sub('''    unordered_set<int> s1(R.adjL[l1].begin(), R.adjL[l1].end());''',
    '''    unordered_set<int> s1;
    for (int e : R.adjL[l1]) s1.insert(ID(e));''')
sub('for (int u : R.adjL[l2]) if (s1.count(u)) out.push_back(u);',
    'for (int e : R.adjL[l2]) if (s1.count(ID(e))) out.push_back(ID(e));')

out = s.replace('\n', '\r\n') if crlf else s
io.open(dst, 'w', encoding='utf-8', newline='').write(out)
print('wrote %s (%s, %s)' % (dst, 'omp' if is_omp else 'serial', 'crlf' if crlf else 'lf'))
