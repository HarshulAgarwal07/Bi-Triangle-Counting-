// konect_fast.cpp
// Blazingly fast KONECT bipartite -> signed bipartite converter.
//
// Same semantics as the original Python script:
//   - Skips blank lines and lines starting with '%'
//   - Remaps src/tgt tokens to strictly contiguous 0-based ids,
//     in order of first appearance (left partition = col 1, right = col 2)
//   - If a 3rd column exists AND it's not exactly "1"/"0" AND it isn't a
//     plain non-negative integer, it's parsed as a float weight:
//        weight > 0 -> sign 0 (positive), else sign 1 (negative)
//     Otherwise falls back to default_sign, or random injection if enabled.
//   - Duplicate (l,u) edges: last occurrence wins (matches Python dict overwrite)
//
// Build:
//   g++ -O3 -march=native -flto -std=c++17 -fopenmp -o konect_fast konect_fast.cpp
//   (drop -fopenmp if you don't have it; code still works single-threaded)
//
// Usage:
//   ./konect_fast <input_path> <output_path> [default_sign] [inject_neg_prob]
//   input_path may end in .bz2 (decompressed via `bzcat` pipe)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::steady_clock;

// ---------------- mmap ----------------
struct MappedFile {
    char* data = nullptr;
    size_t size = 0;
    int fd = -1;
    bool ok() const { return data != nullptr; }

    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept : data(o.data), size(o.size), fd(o.fd) {
        o.data = nullptr; o.size = 0; o.fd = -1;
    }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) {
            if (data) munmap(data, size);
            if (fd >= 0) close(fd);
            data = o.data; size = o.size; fd = o.fd;
            o.data = nullptr; o.size = 0; o.fd = -1;
        }
        return *this;
    }
    ~MappedFile() {
        if (data) munmap(data, size);
        if (fd >= 0) close(fd);
    }
};

static MappedFile map_file(const char* path) {
    MappedFile mf;
    mf.fd = open(path, O_RDONLY);
    if (mf.fd < 0) return mf;
    struct stat st;
    if (fstat(mf.fd, &st) != 0) { close(mf.fd); mf.fd = -1; return mf; }
    mf.size = (size_t)st.st_size;
    if (mf.size == 0) return mf;
    mf.data = (char*)mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, mf.fd, 0);
    if (mf.data == MAP_FAILED) { mf.data = nullptr; return mf; }
    madvise(mf.data, mf.size, MADV_SEQUENTIAL);
    return mf;
}

// ---------------- bz2 via bzcat pipe ----------------
static vector<char> decompress_bz2(const char* path) {
    string cmd = "bzcat \"" + string(path) + "\"";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { fprintf(stderr, "Failed to run bzcat. Is bzip2 installed?\n"); exit(1); }
    vector<char> buf;
    buf.reserve(1u << 28); // 256MB initial guess, grows as needed
    const size_t CH = 1 << 20;
    size_t off = 0;
    while (true) {
        buf.resize(off + CH);
        size_t got = fread(buf.data() + off, 1, CH, pipe);
        off += got;
        if (got < CH) break;
    }
    buf.resize(off);
    pclose(pipe);
    return buf;
}

// ---------------- tokenizing ----------------
struct Token { const char* p; uint32_t len; };

static inline bool all_digits(const char* p, uint32_t len) {
    if (len == 0) return false;
    for (uint32_t i = 0; i < len; i++) if (p[i] < '0' || p[i] > '9') return false;
    return true;
}
static inline bool tok_eq(const char* p, uint32_t len, const char* lit) {
    size_t ll = strlen(lit);
    return len == ll && memcmp(p, lit, ll) == 0;
}

struct ParsedLine {
    Token src, tgt;
    int8_t sign; // -1 = "no numeric weight found, use default/random fallback"
};

static void parse_chunk(const char* buf, size_t begin, size_t end, vector<ParsedLine>& out) {
    size_t i = begin;
    while (i < end) {
        size_t line_start = i;
        while (i < end && buf[i] != '\n') i++;
        size_t line_end = i;
        if (i < end) i++; // consume '\n'

        size_t ls = line_start, le = line_end;
        while (le > ls && (buf[le-1] == '\r' || buf[le-1] == ' ' || buf[le-1] == '\t')) le--;
        while (ls < le && (buf[ls] == ' ' || buf[ls] == '\t')) ls++;

        if (ls >= le) continue;
        if (buf[ls] == '%') continue;

        Token toks[3]; int ntok = 0;
        size_t p = ls;
        while (p < le && ntok < 3) {
            while (p < le && (buf[p] == ' ' || buf[p] == '\t')) p++;
            if (p >= le) break;
            size_t ts = p;
            while (p < le && !(buf[p] == ' ' || buf[p] == '\t')) p++;
            toks[ntok++] = Token{ buf + ts, (uint32_t)(p - ts) };
        }
        if (ntok < 2) continue;

        ParsedLine pl;
        pl.src = toks[0];
        pl.tgt = toks[1];
        pl.sign = -1;

        if (ntok >= 3) {
            const char* wp = toks[2].p; uint32_t wl = toks[2].len;
            bool is_one_or_zero = tok_eq(wp, wl, "1") || tok_eq(wp, wl, "0");
            bool digits_only = all_digits(wp, wl);
            if (!is_one_or_zero && !digits_only) {
                char tmp[64];
                uint32_t cl = wl < 63 ? wl : 63;
                memcpy(tmp, wp, cl); tmp[cl] = 0;
                char* endp = nullptr;
                double weight = strtod(tmp, &endp);
                // Require the WHOLE token to be consumed, matching Python's
                // float(x) which raises ValueError on any trailing garbage
                // (e.g. strtod happily parses "nan_bad" as NaN; Python doesn't).
                if (endp == tmp + cl) pl.sign = (weight > 0) ? 0 : 1;
            }
        }
        out.push_back(pl);
    }
}

static vector<size_t> compute_chunk_bounds(const char* buf, size_t size, int nchunks) {
    vector<size_t> bounds(nchunks + 1);
    bounds[0] = 0;
    for (int i = 1; i < nchunks; i++) {
        size_t approx = size * (size_t)i / (size_t)nchunks;
        while (approx < size && buf[approx] != '\n') approx++;
        if (approx < size) approx++;
        bounds[i] = approx;
    }
    bounds[nchunks] = size;
    return bounds;
}

// good integer hash for the (l,u) edge key so we don't rely on identity hashing
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
static inline uint64_t hash_sv(string_view sv) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : sv) { h ^= c; h *= 1099511628211ULL; }
    return mix64(h);
}
static inline size_t next_pow2(size_t n) {
    size_t p = 16;
    while (p < n) p <<= 1;
    return p;
}

// Open-addressing hashmap: string_view -> int32_t id.
// std::unordered_map is node-based (per-entry heap alloc + pointer chasing);
// for tens of millions of ids that dominates runtime. This is flat, cache-friendly.
struct IdMap {
    struct Slot { string_view key; int32_t val; bool used = false; };
    vector<Slot> table;
    size_t count = 0, mask = 0;

    void init(size_t cap_pow2) { table.assign(cap_pow2, Slot{}); mask = cap_pow2 - 1; count = 0; }

    void grow() {
        vector<Slot> old = std::move(table);
        table.assign(old.size() * 2, Slot{});
        mask = table.size() - 1;
        for (auto& s : old) if (s.used) insert_no_check(s.key, s.val);
    }
    void insert_no_check(string_view k, int32_t v) {
        size_t idx = hash_sv(k) & mask;
        while (table[idx].used) idx = (idx + 1) & mask;
        table[idx] = Slot{ k, v, true };
    }
    // returns the id, assigning a new one (next_id = current count) if absent
    int32_t get_or_assign(string_view k) {
        if ((count + 1) * 10 >= table.size() * 7) grow(); // load factor ~0.7
        size_t idx = hash_sv(k) & mask;
        while (table[idx].used) {
            if (table[idx].key == k) return table[idx].val;
            idx = (idx + 1) & mask;
        }
        int32_t v = (int32_t)count;
        table[idx] = Slot{ k, v, true };
        count++;
        return v;
    }
    size_t size() const { return count; }
};

// Open-addressing hashmap: uint64_t (l,u) -> uint8_t sign, overwrite semantics.
struct EdgeMap {
    struct Slot { uint64_t key; uint8_t val; bool used = false; };
    vector<Slot> table;
    size_t count = 0, mask = 0;

    void init(size_t cap_pow2) { table.assign(cap_pow2, Slot{}); mask = cap_pow2 - 1; count = 0; }

    void grow() {
        vector<Slot> old = std::move(table);
        table.assign(old.size() * 2, Slot{});
        mask = table.size() - 1;
        for (auto& s : old) if (s.used) insert_no_check(s.key, s.val);
    }
    void insert_no_check(uint64_t k, uint8_t v) {
        size_t idx = mix64(k) & mask;
        while (table[idx].used) idx = (idx + 1) & mask;
        table[idx] = Slot{ k, v, true };
    }
    void set(uint64_t k, uint8_t v) { // last-write-wins, like Python dict[k] = v
        if ((count + 1) * 10 >= table.size() * 7) grow();
        size_t idx = mix64(k) & mask;
        while (table[idx].used) {
            if (table[idx].key == k) { table[idx].val = v; return; }
            idx = (idx + 1) & mask;
        }
        table[idx] = Slot{ k, v, true };
        count++;
    }
    size_t size() const { return count; }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_path> <output_path> [default_sign] [inject_neg_prob]\n", argv[0]);
        return 1;
    }
    string input_path = argv[1];
    string output_path = argv[2];
    int default_sign = argc > 3 ? atoi(argv[3]) : 0;
    double inject_neg_prob = argc > 4 ? atof(argv[4]) : 0.0;

    auto t0 = Clock::now();

    vector<char> bz_buf;
    const char* buf; size_t size;
    MappedFile mf;
    bool is_bz2 = input_path.size() > 4 && input_path.compare(input_path.size() - 4, 4, ".bz2") == 0;
    if (is_bz2) {
        fprintf(stderr, "Decompressing %s via bzcat...\n", input_path.c_str());
        bz_buf = decompress_bz2(input_path.c_str());
        buf = bz_buf.data(); size = bz_buf.size();
    } else {
        mf = map_file(input_path.c_str());
        if (!mf.ok()) { fprintf(stderr, "Failed to open/mmap %s\n", input_path.c_str()); return 1; }
        buf = mf.data; size = mf.size;
    }
    fprintf(stderr, "Loaded %.2f MB in %.3fs\n", size / 1e6, chrono::duration<double>(Clock::now() - t0).count());

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    int nchunks = max(1, nthreads);
    auto bounds = compute_chunk_bounds(buf, size, nchunks);
    vector<vector<ParsedLine>> chunk_results(nchunks);

    auto t1 = Clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < nchunks; c++) {
        chunk_results[c].reserve((bounds[c+1] - bounds[c]) / 20 + 16);
        parse_chunk(buf, bounds[c], bounds[c+1], chunk_results[c]);
    }
    fprintf(stderr, "Tokenized in %.3fs (%d thread(s))\n",
            chrono::duration<double>(Clock::now() - t1).count(), nchunks);

    // ---- sequential merge: id assignment must preserve first-appearance order ----
    auto t2 = Clock::now();
    size_t total_lines = 0;
    for (auto& v : chunk_results) total_lines += v.size();

    // Size tables up front so we never pay for a mid-run rehash-and-grow
    // (the single most expensive thing an open-addressing table can do),
    // while keeping tables small enough to stay cache/TLB-friendly.
    // ~1.6x the worst-case count keeps load factor safely under the 0.7
    // grow threshold without over-allocating (over-sizing hurts more than
    // an occasional grow does, since probing a huge sparse table blows
    // the cache regardless of collision rate).
    size_t cap = next_pow2((size_t)(total_lines * 1.6) + 64);
    IdMap left_to_id, right_to_id;
    left_to_id.init(cap);
    right_to_id.init(cap);

    EdgeMap edges;
    edges.init(cap);

    mt19937_64 rng(12345);
    uniform_real_distribution<double> unif(0.0, 1.0);

    for (int c = 0; c < nchunks; c++) {
        for (auto& pl : chunk_results[c]) {
            string_view sv_src(pl.src.p, pl.src.len);
            string_view sv_tgt(pl.tgt.p, pl.tgt.len);

            int32_t l = left_to_id.get_or_assign(sv_src);
            int32_t u = right_to_id.get_or_assign(sv_tgt);

            uint8_t sign;
            if (pl.sign >= 0) sign = (uint8_t)pl.sign;
            else if (inject_neg_prob > 0.0) sign = (unif(rng) < inject_neg_prob) ? 1 : 0;
            else sign = (uint8_t)default_sign;

            uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)u;
            edges.set(key, sign); // last write wins -> matches Python dict overwrite
        }
        vector<ParsedLine>().swap(chunk_results[c]); // free memory as we go
    }
    fprintf(stderr, "Merged/deduped in %.3fs\n", chrono::duration<double>(Clock::now() - t2).count());

    int32_t nL = (int32_t)left_to_id.size();
    int32_t nU = (int32_t)right_to_id.size();
    size_t nE = edges.size();

    // ---- fast buffered output ----
    auto t3 = Clock::now();
    FILE* out = fopen(output_path.c_str(), "wb");
    if (!out) { fprintf(stderr, "Failed to open output file %s\n", output_path.c_str()); return 1; }

    const size_t OUTBUF = 1u << 24; // 16MB
    vector<char> obuf(OUTBUF);
    size_t opos = 0;
    auto flush = [&]() { if (opos) { fwrite(obuf.data(), 1, opos, out); opos = 0; } };
    auto write_uint = [&](uint32_t v) {
        char tmp[12]; int tl = 0;
        if (v == 0) tmp[tl++] = '0';
        while (v > 0) { tmp[tl++] = char('0' + (v % 10)); v /= 10; }
        if (opos + (size_t)tl > OUTBUF) flush();
        for (int i = tl - 1; i >= 0; i--) obuf[opos++] = tmp[i];
    };
    auto write_char = [&](char ch) { if (opos + 1 > OUTBUF) flush(); obuf[opos++] = ch; };

    write_uint((uint32_t)nL); write_char(' ');
    write_uint((uint32_t)nU); write_char(' ');
    write_uint((uint32_t)nE); write_char('\n');

    for (auto& slot : edges.table) {
        if (!slot.used) continue;
        uint32_t l = (uint32_t)(slot.key >> 32);
        uint32_t u = (uint32_t)(slot.key & 0xffffffffu);
        write_uint(l); write_char(' ');
        write_uint(u); write_char(' ');
        write_uint(slot.val); write_char('\n');
    }
    flush();
    fclose(out);

    fprintf(stderr, "Wrote output in %.3fs\n", chrono::duration<double>(Clock::now() - t3).count());
    fprintf(stderr, "TOTAL: %.3fs | nL=%d nU=%d nE=%zu\n",
            chrono::duration<double>(Clock::now() - t0).count(), nL, nU, nE);
    return 0;
}
