#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <thread>
#include <stdexcept>

using namespace std;
using ll=long long;
#define OVERLOAD_REP(_1, _2, _3, name, ...) name
#define REP1(i, n) for (auto i = decay_t<decltype(n)>{}; (i) != (n); ++(i))
#define REP2(i, l, r) for (auto i = (l); (i) != (r); ++(i))
#define rep(...) OVERLOAD_REP(__VA_ARGS__, REP2, REP1)(__VA_ARGS__)
#define INF 9223372036854775807
#define mod(n, r) ((n) % (r) + (r)) % (r)
template<typename T>
std::vector<T> create_copy(std::vector<T> const &vec) { std::vector<T> v(vec); return v;}

ll mns(ll i, ll j) {
    return i - j;
}

ll add(ll i, ll j) {
    return i + j;
}

ll add(ll i){
    return add(i, 1);
}

void inner_add(ll &i){
    i = i + 1;
}

// dst: std::set or std::map
// begin & end: iterator of sorted items
template<class Tree, class Iterator>
inline void insert_sorted_sequence(Tree& dst, Iterator begin, Iterator end)
{
    auto pos = dst.begin();
    for (auto it = begin; it != end; ++it) {
        pos = dst.emplace_hint(pos, *it);
        ++pos;
    }
}

template<typename T>
void symmetricDifference(unordered_set<T>& A, const unordered_set<T>& B) {
    for (const auto& element : B) {
        // Check if the element is not in A
        if (A.find(element) == A.end()) {
            // If not in A, add it to A
            A.insert(element);
        } else {
            // If already in A, remove it from A
            A.erase(element);
        }
    }
}

typedef unsigned int pt_idx_t;

struct DSU {
    public:
        // DSU() : _n(0) {}
        explicit DSU(int n) : _n(n), num_groups(n), parent_or_size(n, -1){
            rep(i, n) ph_root.push_back(i);
        }

        pair<bool, pt_idx_t>  merge(pt_idx_t a, pt_idx_t b) {
            if (num_groups == 1) return make_pair(false, a);
            pair<bool, pt_idx_t> ret = make_pair(false, a);
            assert(0 <= a && a < _n);
            assert(0 <= b && b < _n);
            int x = leader(a), y = leader(b);
            if (x == y) return ret;
            if (ph_root[x] < ph_root[y]) {
                ret = make_pair(true, ph_root[x]);
                ph_root[x] = ph_root[y];
            } else if (ph_root[x] > ph_root[y]) {
                ret = make_pair(true, ph_root[y]);
                ph_root[y] = ph_root[x];
            } else assert (false);
            if (-parent_or_size[x] < -parent_or_size[y]) swap(x, y);
            parent_or_size[x] += parent_or_size[y];
            parent_or_size[y] = x;
            num_groups--;
            return ret;
        }

        bool same(int a, int b) {
            assert(0 <= a && a < _n);
            assert(0 <= b && b < _n);
            return leader(a) == leader(b);
        }

        int leader(int a) {
            assert(0 <= a && a < _n);
            if (parent_or_size[a] < 0) return a;
            return parent_or_size[a] = leader(parent_or_size[a]);
        }

        int size(int a) {
            assert(0 <= a && a < _n);
            return -parent_or_size[leader(a)];
        }

        vector<vector<int>> groups() {
            vector<int> leader_buf(_n), group_size(_n);
            for (int i = 0; i < _n; i++) {
                leader_buf[i] = leader(i);
                group_size[leader_buf[i]]++;
            }
            vector<vector<int>> result(_n);
            for (int i = 0; i < _n; i++) {
                result[i].reserve(group_size[i]);
            }
            for (int i = 0; i < _n; i++) {
                result[leader_buf[i]].push_back(i);
            }
            result.erase(
                remove_if(result.begin(), result.end(),
                            [&](const vector<int>& v) { return v.empty(); }),
                result.end());
            return result;
        }

        int get_ph_root(int a) {
            assert(0 <= a && a < _n);
            return ph_root[leader(a)];
        }

    private:
        int _n;
        int num_groups;
        // root node: -1 * component size
        // otherwise: parent
        vector<int> parent_or_size;
        vector<int> ph_root; 
};

typedef double diameter_t;
typedef unsigned int dim_t;
typedef unsigned long long simp_idx_t;

struct DiameterEntry {
    dim_t dim;
    simp_idx_t idx;
    diameter_t diameter;
    DiameterEntry(dim_t dim, simp_idx_t idx, diameter_t &diameter) : dim(dim), idx(idx), diameter(diameter) {}
    DiameterEntry() : dim(0), idx(0), diameter(0) {} 
};

struct DiameterEntryGreaterDiameterSmallerIndex {
    bool operator()(const DiameterEntry& a, const DiameterEntry& b) {
        return (a.diameter > b.diameter) || (a.diameter == b.diameter && a.idx < b.idx);
    }
};

struct DiameterEntrySmallerDiameterGreaterIndex {
    bool operator()(const DiameterEntry& a, const DiameterEntry& b) {
        return (a.diameter < b.diameter) || (a.diameter == b.diameter && a.idx > b.idx);
    }
};

typedef priority_queue<DiameterEntry, vector<DiameterEntry>, DiameterEntryGreaterDiameterSmallerIndex> Row;
typedef priority_queue<DiameterEntry, vector<DiameterEntry>, DiameterEntrySmallerDiameterGreaterIndex> Column;
typedef unordered_map<simp_idx_t, simp_idx_t> DeathToBirth;
typedef unordered_map<simp_idx_t, simp_idx_t> BirthToDeath;
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> LeftReducingMatrix; 
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> RightReducingMatrix;
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> InvLeftReducingMatrix; 
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> InvRightReducingMatrix; 

struct RipsPersistentHomology {
    public:
        pt_idx_t N;
        vector<vector<diameter_t>> dist;
        dim_t maxdim;
        vector<DeathToBirth> death_to_birth;
        vector<BirthToDeath> birth_to_death;
        LeftReducingMatrix W;
        InvLeftReducingMatrix invW;
        RightReducingMatrix V;
        InvRightReducingMatrix invV;
        vector<vector<simp_idx_t>> binomial_table;
        diameter_t enclosing_radius;
        unsigned long long num_threads;
        RipsPersistentHomology(vector<vector<diameter_t>> dist, dim_t maxdim, unsigned long long num_threads=1ULL) : N(dist.size()), dist(dist), maxdim(maxdim), num_threads(num_threads){
            rep(i, maxdim+1) death_to_birth.push_back(DeathToBirth());
            rep(i, maxdim+1) birth_to_death.push_back(BirthToDeath());
            rep(i, maxdim+1) W.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            rep(i, maxdim+1) invW.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            rep(i, maxdim+1) V.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            rep(i, maxdim+1) invV.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            rep(i, N + 1) {
                vector<simp_idx_t> binom_i;
                binom_i.push_back(1); // j = 0
                rep(j, 1, maxdim + 3) {
                    simp_idx_t val;
                    if (i == 0) val = 0;
                    else val = binomial_table[i - 1][j - 1] + binomial_table[i - 1][j];
                    binom_i.push_back(val); // j
                }
                binomial_table.push_back(binom_i);
            }
            diameter_t M = 0;
            rep(j, N) M = max(M, dist[0][j]);
            enclosing_radius = M;
            rep(i, 1, N){
                M = 0;
                rep(j, N) M = max(M, dist[i][j]);
                enclosing_radius = min(enclosing_radius, M);
            }
        }

        simp_idx_t binomial(pt_idx_t n, dim_t r) { 
            if (n < r || r < 0) return 0;
            return binomial_table[n][r];
        }
        
        simp_idx_t get_simplex_index(vector<pt_idx_t> &vertex_list) {
            dim_t dim = vertex_list.size() - 1;
            simp_idx_t ret = 0;
            rep(i, dim+1) {
                ret += binomial(vertex_list[i], dim - i + 1);
            }
            return ret;
        }

        pt_idx_t get_max_vertex(dim_t dim, simp_idx_t idx, pt_idx_t right) {
            pt_idx_t left = dim, mid;
            while (right - left > 1) {
                mid = (left + right) / 2;
                if (binomial(mid, dim + 1) <= idx) left = mid;
                else right = mid;
            }
            return left;
        }

        vector<pt_idx_t> get_simplex_vertices(dim_t dim, simp_idx_t idx) {
            vector<pt_idx_t> ret; 
            pt_idx_t max_vertex = N;
            int _dim = dim, _idx = idx;
            while (ret.size() <= dim) {
                max_vertex = get_max_vertex(_dim, _idx, max_vertex);
                ret.push_back(max_vertex);
                _idx -= binomial(max_vertex, _dim+1);
                _dim -= 1;
            }
            return ret;
        }

        diameter_t get_diameter(dim_t dim, simp_idx_t idx) {
            vector<pt_idx_t> vertex_list = get_simplex_vertices(dim, idx);
            diameter_t ret = 0;
            rep(i, dim+1) rep(j, i+1, dim+1) {
                ret = max(ret, dist[vertex_list[i]][vertex_list[j]]);
            }
            return ret;
        }

        pair<pt_idx_t, pt_idx_t> get_max_edge(dim_t dim, simp_idx_t idx) {
            vector<pt_idx_t> vertex_list = get_simplex_vertices(dim, idx);
            pt_idx_t v1=0, v2=0;
            diameter_t max_dist = 0;
            rep(i, dim+1) rep(j, i+1, dim+1) {
                if (max_dist < dist[vertex_list[i]][vertex_list[j]]) {
                    max_dist = dist[vertex_list[i]][vertex_list[j]];
                    v1 = vertex_list[i];
                    v2 = vertex_list[j];
                }
            }
            return make_pair(v1, v2);
        }

        pair<bool, simp_idx_t> heappush_coboundary(Row &row, dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt, bool push=true) {
            vector<pt_idx_t> vertex_list = get_simplex_vertices(dim, idx); // length: dim+1
            int k = dim;
            simp_idx_t term_1 = 0;
            simp_idx_t term_2 = 0;
            rep(i, dim+1) term_2 += binomial(vertex_list[dim-i], i+1);
            rep(i, N) {
                int j = N - i - 1;
                if (k >= 0 && j == vertex_list[dim-k]){
                    if (k >= 0) term_2 -= binomial(vertex_list[dim-k], k+1);
                    else term_2 = 0;
                    k -= 1;
                    term_1 += binomial(vertex_list[dim-k-1], k+3);
                    continue;
                }
                simp_idx_t coface_idx = term_1 + binomial(j, k+2) + term_2;
                diameter_t coface_diameter = get_diameter(dim+1, coface_idx);
                if (encl_opt && coface_diameter > enclosing_radius) {
                    continue;
                }
                if (emg_opt && coface_diameter == get_diameter(dim, idx)) {
                    emg_opt = false;
                    if (death_to_birth[dim].count(coface_idx) == 0) {
                        return make_pair(true, coface_idx); 
                    }
                }
                if (push) {
                    row.push(DiameterEntry(dim+1, coface_idx, coface_diameter));
                }
            }
            if (row.empty()) {
                return make_pair(false, 0);
            } else {
                return make_pair(false, row.top().idx);
            }
        }
        pair<bool, simp_idx_t> heappush_coboundary(dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt) {
            Row row; row.push(DiameterEntry());
            return heappush_coboundary(row, dim, idx, encl_opt, emg_opt, false);
        }

        pair<bool, simp_idx_t> heappush_boundary(Column &column, dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt, bool push=true) {
            vector<pt_idx_t> vertex_list = get_simplex_vertices(dim, idx); // length: dim+1
            simp_idx_t term_1 = 0, term_2 = 0;
            rep(i, dim+1){
                int k = dim - i;
                if (k == dim) {
                    rep(j, dim) term_2 += binomial(vertex_list[dim-j], j+1);
                } else {
                    term_1 += binomial(vertex_list[dim-k-1], k+1);
                    term_2 -= binomial(vertex_list[dim-k], k+1);
                }
                simp_idx_t face_idx = term_1 + term_2;
                diameter_t face_diameter = get_diameter(dim-1, face_idx);
                if (encl_opt && face_diameter > enclosing_radius) {
                    continue;
                }
                if (emg_opt && face_diameter == get_diameter(dim, idx)){
                    emg_opt = false;
                    if (birth_to_death[dim-1].count(face_idx) == 0) {
                        return make_pair(true, face_idx);
                    }
                }
                if (push) {
                    column.push(DiameterEntry(dim-1, face_idx, face_diameter));
                }
            }
            if (column.empty()) {
                throw runtime_error("Unexpected empty column encountered in heappush_boundary. This is a bug in the C++ implementation.");
            } else {
                return make_pair(false, column.top().idx);
            }
        }
        pair<bool, simp_idx_t> heappush_boundary(dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt) {
            Column column; column.push(DiameterEntry());
            return heappush_boundary(column, dim, idx, encl_opt, emg_opt, false);
        }

        void check_zero_apparent(dim_t dim, simp_idx_t idx, bool encl_opt, unsigned int &is_apparent, simp_idx_t &apparent_cofacet_idx) {
            pair<bool, simp_idx_t> zero_pivot_cofacet = heappush_coboundary(dim, idx, encl_opt, true);
            if (!zero_pivot_cofacet.first) return;
            pair<bool, simp_idx_t> zero_pivot_face = heappush_boundary(dim+1, zero_pivot_cofacet.second, encl_opt, true);
            if (!zero_pivot_face.first) return;
            if (zero_pivot_face.second == idx) {
                is_apparent = 1;
                apparent_cofacet_idx = zero_pivot_cofacet.second;
            }
            return;
        }

        void compute_ph(bool enclosing_opt=true, bool emergent_opt=true, bool clearing_opt=true, bool get_inv=false) {
            rep(_dim, 1, max(2U, maxdim+1)) {
                dim_t dim = _dim;
                vector<DiameterEntry> simplex_list;
                rep(i, binomial(N, dim+1)){
                    simp_idx_t idx = i;
                    if (clearing_opt && death_to_birth[dim-1].count(idx)) continue; 
                    diameter_t diameter = get_diameter(dim, i);
                    if (enclosing_opt && (diameter > enclosing_radius)) continue;
                    simplex_list.push_back(DiameterEntry(dim, idx, diameter));
                }
                vector<DiameterEntry> sorted_simplex_list;
                if (maxdim > 0) {
                    vector<unsigned int> is_apparent_list(simplex_list.size());
                    vector<simp_idx_t> apparent_cofacet_list(simplex_list.size());
                    vector<thread> apparent_threads;
                    unsigned long long thread_loops = simplex_list.size() / num_threads + ((simplex_list.size() % num_threads) >= 1); 
                    rep(i, num_threads){
                        apparent_threads.push_back(thread([&]{
                            rep(j, thread_loops * i, thread_loops * (i+1)){
                                if (j >= simplex_list.size()) break; 
                                simp_idx_t idx = simplex_list[j].idx; 
                                check_zero_apparent(dim, idx, enclosing_opt, is_apparent_list[j], apparent_cofacet_list[j]);
                            }
                        }));
                    }
                    for (thread &t: apparent_threads) t.join();
                    rep(i, simplex_list.size()) { 
                        if (is_apparent_list[i]) {
                            simp_idx_t idx = simplex_list[i].idx;
                            death_to_birth[dim][apparent_cofacet_list[i]] = idx; 
                            W[dim][idx].insert(idx); invW[dim][idx].insert(idx);
                        }
                        else sorted_simplex_list.push_back(simplex_list[i]);
                    }
                } else {
                    sorted_simplex_list = create_copy(simplex_list);
                }
                sort(sorted_simplex_list.begin(), sorted_simplex_list.end(), DiameterEntryGreaterDiameterSmallerIndex());
                if (dim == 1){
                    DSU dsu(N);
                    vector<DiameterEntry> _sorted_simplex_list; 
                    rep(_simp_entr, sorted_simplex_list.rbegin(), sorted_simplex_list.rend()){ 
                        vector<pt_idx_t> vertex_list = get_simplex_vertices(1, _simp_entr->idx);
                        pair<bool, pt_idx_t> dsu_ret = dsu.merge(vertex_list[0], vertex_list[1]);
                        if (dsu_ret.first) death_to_birth[0][_simp_entr->idx] = dsu_ret.second;
                        else _sorted_simplex_list.push_back(*_simp_entr);
                    }
                    sorted_simplex_list = _sorted_simplex_list;
                    reverse(sorted_simplex_list.begin(), sorted_simplex_list.end());
                }
                if (maxdim==0) break;
                rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){
                    W[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    invW[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    Row row;
                    pair<bool, simp_idx_t> ret = heappush_coboundary(row, dim, _simp_entr->idx, enclosing_opt, emergent_opt);
                    if (ret.first) {
                        death_to_birth[dim][ret.second] = _simp_entr->idx;
                        continue;
                    }
                    while (clearing_opt || !row.empty()){ 
                        // when clearing_opt == true, `row` is expected to be non-empty
                        if (row.empty()) {
                            throw runtime_error("Unexpected empty row encountered in compute_ph. This is a bug in the C++ implementation.");
                        }

                        DiameterEntry pivot_entry = row.top(); 
                        row.pop(); 

                        if (!row.empty() && pivot_entry.idx == row.top().idx){
                            row.pop(); 
                        } else if (death_to_birth[dim].count(pivot_entry.idx)){
                            simp_idx_t target_simp = death_to_birth[dim][pivot_entry.idx]; 
                            symmetricDifference(W[dim][_simp_entr->idx], W[dim][target_simp]);
                            if (get_inv) {
                                invW[dim][target_simp].insert(_simp_entr->idx); 
                            }
                            rep(sweep_idx, W[dim][target_simp].begin(), W[dim][target_simp].end()) {
                                heappush_coboundary(row, dim, *sweep_idx, enclosing_opt, false);
                            }
                            row.push(pivot_entry);
                        } else {
                            death_to_birth[dim][pivot_entry.idx] = _simp_entr->idx;
                            break;
                        }
                    }
                }
            }
        }

        void compute_ph_right(bool enclosing_opt=true, bool emergent_opt=true, bool get_inv=false) {
            rep(_dim, maxdim+1) {
                dim_t dim = _dim;
                if (dim >= 1){
                    vector<DiameterEntry> lower_simplex_list;
                    rep(i, binomial(N, dim+1)){
                        simp_idx_t idx = i; 
                        diameter_t diameter = get_diameter(dim, i);
                        if (enclosing_opt && (diameter > enclosing_radius)) continue;
                        lower_simplex_list.push_back(DiameterEntry(dim, idx, diameter));
                    }
                    vector<unsigned int> is_apparent_list(lower_simplex_list.size()); 
                    vector<simp_idx_t> apparent_cofacet_list(lower_simplex_list.size());
                    vector<thread> apparent_threads;
                    unsigned long long thread_loops = lower_simplex_list.size() / num_threads + ((lower_simplex_list.size() % num_threads) >= 1);
                    rep(i, num_threads){
                        apparent_threads.push_back(thread([&]{
                            rep(j, thread_loops * i, thread_loops * (i+1)){
                                if (j >= lower_simplex_list.size()) break; 
                                simp_idx_t idx = lower_simplex_list[j].idx; 
                                check_zero_apparent(dim, idx, enclosing_opt, is_apparent_list[j], apparent_cofacet_list[j]);
                            }
                        }));
                    }
                    for (thread &t: apparent_threads) t.join();
                    rep(i, lower_simplex_list.size()) { 
                        if (is_apparent_list[i]) {
                            simp_idx_t idx = lower_simplex_list[i].idx;
                            simp_idx_t cofacet_idx = apparent_cofacet_list[i];
                            birth_to_death[dim][idx] = cofacet_idx;
                            V[dim][cofacet_idx].insert(cofacet_idx); invV[dim][cofacet_idx].insert(cofacet_idx);
                        }
                    }
                }
                vector<DiameterEntry> sorted_simplex_list;
                rep(i, binomial(N, dim+2)){
                    simp_idx_t idx = i;
                    if (V[dim].count(idx)) continue;
                    diameter_t diameter = get_diameter(dim+1, i);
                    if (enclosing_opt && (diameter > enclosing_radius)) continue;
                    sorted_simplex_list.push_back(DiameterEntry(dim+1, idx, diameter));
                }
                sort(sorted_simplex_list.begin(), sorted_simplex_list.end(), DiameterEntrySmallerDiameterGreaterIndex());
                if (dim == 0){
                    DSU dsu(N);
                    rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){ 
                        vector<pt_idx_t> vertex_list = get_simplex_vertices(1, _simp_entr->idx);
                        pair<bool, pt_idx_t> dsu_ret = dsu.merge(vertex_list[0], vertex_list[1]);
                        if (dsu_ret.first) birth_to_death[0][dsu_ret.second] = _simp_entr->idx;
                    }
                    continue;
                }
                rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){
                    V[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    invV[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    Column column;
                    pair<bool, simp_idx_t> ret = heappush_boundary(column, dim+1, _simp_entr->idx, enclosing_opt, emergent_opt);
                    if (ret.first) {
                        birth_to_death[dim][ret.second] = _simp_entr->idx;
                        continue;
                    }
                    while (!column.empty()){ 
                        DiameterEntry pivot_entry = column.top(); column.pop(); 
                        if (!column.empty() && pivot_entry.idx == column.top().idx){
                            column.pop(); 
                        } else if (birth_to_death[dim].count(pivot_entry.idx)){
                            simp_idx_t target_simp = birth_to_death[dim][pivot_entry.idx]; 
                            symmetricDifference(V[dim][_simp_entr->idx], V[dim][target_simp]);
                            if (get_inv) {
                                invV[dim][target_simp].insert(_simp_entr->idx);
                            }
                            rep(sweep_idx, V[dim][target_simp].begin(), V[dim][target_simp].end()) {
                                heappush_boundary(column, dim+1, *sweep_idx, enclosing_opt, false);
                            }
                            column.push(pivot_entry);
                        } else {
                            birth_to_death[dim][pivot_entry.idx] = _simp_entr->idx;
                            break;
                        }
                    }
                }
            }
        }
};