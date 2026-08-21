# Prototype 2, layered on the packed-adjacency prototype: merge the S and T
# maps into ONE map with a 4-field value. Today the inner loop probes S[y]
# and T[y] separately, and the correction loop probes T.at(y) a third time.
import io, sys

src = sys.argv[1] if len(sys.argv) > 1 else 'v_packed.cpp'
dst = sys.argv[2] if len(sys.argv) > 2 else 'v_st.cpp'
s = io.open(src, encoding='utf-8', newline='').read()

def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "expected %d got %d for: %r" % (count, n, old[:70])
    s = s.replace(old, new)

# ---- one value type holding both the wedge counts and the phi-weighted sums ----
sub('''struct VertexScratch {
    unordered_map<int, pair<ll,ll>> S; // y -> (even,odd) wedge counts
    unordered_map<int, pair<ll,ll>> T; // y -> (even,odd) sum of phi(sigma(v),x)
    unordered_map<int, pair<ll,ll>> H; // z -> (even,odd) super-wedge counts
};''',
'''struct Wedge {
    ll se = 0, so = 0;   // wedge counts, by parity   (was S)
    ll te = 0, to = 0;   // sum of phi(sigma(v),x)    (was T)
};

struct VertexScratch {
    unordered_map<int, Wedge> S;       // y -> wedge counts AND phi-weighted sums
    unordered_map<int, pair<ll,ll>> H; // z -> (even,odd) super-wedge counts
};''')

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

# ---- inner loop: one probe instead of two ----
sub('''            auto& s = S[y];
            auto& t = T[y];
            if (w == 0) { s.first  += 1; t.first  += phi_v_x; }
            else        { s.second += 1; t.second += phi_v_x; }''',
'''            auto& s = S[y];
            if (w == 0) { s.se += 1; s.te += phi_v_x; }
            else        { s.so += 1; s.to += phi_v_x; }''', 2)

# ---- correction loop: drops the T.at(y) probe entirely ----
sub('''        ll se = kv.second.first, so = kv.second.second;
        ll te = T.at(y).first, to = T.at(y).second;''',
'''        ll se = kv.second.se, so = kv.second.so;
        ll te = kv.second.te, to = kv.second.to;''', 2)

# ---- H build (the two remaining reads of the old pair layout) ----
sub('ll se = kv.second.first, so = kv.second.second;',
    'll se = kv.second.se, so = kv.second.so;', 2)

io.open(dst, 'w', encoding='utf-8', newline='\n').write(s)
print('wrote', dst)
