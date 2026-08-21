# Prototype: pack the edge sign into the top bit of each adjacency entry,
# so the counting loops never touch signOf. Operates on a COPY in /tmp.
import io, sys

src = sys.argv[1] if len(sys.argv) > 1 else '../signed_rswj_count_omp.cpp'
dst = sys.argv[2] if len(sys.argv) > 2 else 'v_packed.cpp'
s = io.open(src, encoding='utf-8', newline='').read().replace('\r\n', '\n')

def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "expected %d got %d for: %r" % (count, n, old[:80])
    s = s.replace(old, new)

# ---- helpers ----
sub('static inline i128 C2(ll n)',
    '''// ---- packed adjacency entry: neighbour id in the low 31 bits, edge sign
// in the top bit. One sequential load gives both, so the counting loops
// never probe signOf at all.
static inline int PACK(int v, int sgn) { return (int)((unsigned)v | ((unsigned)sgn << 31)); }
static inline int ID(int e)  { return e & 0x7fffffff; }
static inline int SG(int e)  { return (int)((unsigned)e >> 31); }

static inline i128 C2(ll n)''')

# ---- binary searches: compare rank of the UNPACKED id ----
sub('if (rankOfU(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;',
    'if (rankOfU(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;')
sub('if (rankOfU(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;',
    'if (rankOfU(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;')
sub('if (rankOfL(lst[mid]) > thresholdRank) hi = mid; else lo = mid + 1;',
    'if (rankOfL(ID(lst[mid])) > thresholdRank) hi = mid; else lo = mid + 1;')
sub('if (rankOfL(lst[mid]) < thresholdRank) lo = mid + 1; else hi = mid;',
    'if (rankOfL(ID(lst[mid])) < thresholdRank) lo = mid + 1; else hi = mid;')

# ---- core build: attach signs while filtering ----
sub('for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(u);',
    'for (int u : G.adjL[l]) if (R.aliveU[u]) R.adjL[l].push_back(PACK(u, R.sign(l, u)));')
sub('for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(l);',
    'for (int l : G.adjU[u]) if (R.aliveL[l]) R.adjU[u].push_back(PACK(l, R.sign(l, u)));')

# ---- rank sort on unpacked ids ----
sub('return R.rankOfU(u1) < R.rankOfU(u2);', 'return R.rankOfU(ID(u1)) < R.rankOfU(ID(u2));')
sub('return R.rankOfL(l1) < R.rankOfL(l2);', 'return R.rankOfL(ID(l1)) < R.rankOfL(ID(l2));')

# ---- processVertexL ----
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

sub('''        for (int z : R.psiL(y, rv)) {             // z in U, rank(z) > rv (threshold = rv, see header note)
            int s_yz = R.sign(y, z);''',
    '''        for (int ez : R.psiL(y, rv)) {            // z in U, rank(z) > rv (threshold = rv, see header note)
            int z = ID(ez), s_yz = SG(ez);''')

sub('''    for (int z : R.adjL[l]) {                     // z in N(v)
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiU(z, rv).size();
        if (corr <= 0) continue;
        int s_lz = R.sign(l, z);''',
    '''    for (int ez : R.adjL[l]) {                    // z in N(v)
        int z = ID(ez), s_lz = SG(ez);
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiU(z, rv).size();
        if (corr <= 0) continue;''')

# ---- processVertexU ----
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

sub('''        for (int z : R.psiU(y, rv)) {             // z in L, rank(z) > rv
            int s_yz = R.sign(z, y);''',
    '''        for (int ez : R.psiU(y, rv)) {            // z in L, rank(z) > rv
            int z = ID(ez), s_yz = SG(ez);''')

sub('''    for (int z : R.adjU[u]) {                     // z in N(v)
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiL(z, rv).size();
        if (corr <= 0) continue;
        int s_uz = R.sign(z, u);''',
    '''    for (int ez : R.adjU[u]) {                    // z in N(v)
        int z = ID(ez), s_uz = SG(ez);
        auto it = H.find(z);
        if (it == H.end()) continue;
        ll corr = R.psiL(z, rv).size();
        if (corr <= 0) continue;''')

# ---- work list ----
sub('for (int u : R.adjL[l]) f += R.coreDegU(u);', 'for (int u : R.adjL[l]) f += R.coreDegU(ID(u));')
sub('for (int l : R.adjU[u]) f += R.coreDegL(l);', 'for (int l : R.adjU[u]) f += R.coreDegL(ID(l));')

# ---- brute force still walks adjacency by id ----
sub('''    unordered_set<int> s1(R.adjL[l1].begin(), R.adjL[l1].end());
    vector<int> out;
    for (int u : R.adjL[l2]) if (s1.count(u)) out.push_back(u);''',
    '''    unordered_set<int> s1;
    for (int e : R.adjL[l1]) s1.insert(ID(e));
    vector<int> out;
    for (int e : R.adjL[l2]) if (s1.count(ID(e))) out.push_back(ID(e));''')

io.open(dst, 'w', encoding='utf-8', newline='\n').write(s)
print('wrote', dst)
