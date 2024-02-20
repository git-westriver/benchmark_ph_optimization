#include <bits/stdc++.h>
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

// 非破壊的な set の結合
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
    // A と B の xor をとる．B は変更されないが，A は変更されることに注意．
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
            // エッジの追加によって連結成分が消える場合は，(true, 対応する birth) を返す
            // 消えない場合は，(false, a) を返す
            if (num_groups == 1) return make_pair(false, a);
            pair<bool, pt_idx_t> ret = make_pair(false, a);
            assert(0 <= a && a < _n);
            assert(0 <= b && b < _n);
            int x = leader(a), y = leader(b);
            if (x == y) return ret;
            // ph_root の更新
            if (ph_root[x] < ph_root[y]) {
                ret = make_pair(true, ph_root[x]);
                ph_root[x] = ph_root[y];
            } else if (ph_root[x] > ph_root[y]) {
                ret = make_pair(true, ph_root[y]);
                ph_root[y] = ph_root[x];
            } else assert (false);
            // parent_or_size の更新
            if (-parent_or_size[x] < -parent_or_size[y]) swap(x, y);
            parent_or_size[x] += parent_or_size[y];
            parent_or_size[y] = x;
            // num_groups の更新
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

typedef double diameter_t; // 注意：かならず pointer を引数にする
typedef unsigned int dim_t;
typedef unsigned long long simp_idx_t;

struct DiameterEntry { // DiameterEntry というより，Simplex では？
    dim_t dim;
    simp_idx_t idx;
    diameter_t diameter;
    DiameterEntry(dim_t dim, simp_idx_t idx, diameter_t &diameter) : dim(dim), idx(idx), diameter(diameter) {}
    DiameterEntry() : dim(0), idx(0), diameter(0) {} // 空のオブジェクト
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
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> LeftReducingMatrix; // 次元，simplex_idx
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> RightReducingMatrix; // 次元，simplex_idx
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> InvLeftReducingMatrix; // 次元，simplex_idx
typedef vector<unordered_map<simp_idx_t, unordered_set<simp_idx_t>>> InvRightReducingMatrix; // 次元，simplex_idx

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
            // death_to_birth の初期化
            rep(i, maxdim+1) death_to_birth.push_back(DeathToBirth());
            // birth_to_death の初期化
            rep(i, maxdim+1) birth_to_death.push_back(BirthToDeath());
            // W の初期化
            rep(i, maxdim+1) W.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            // invW の初期化
            rep(i, maxdim+1) invW.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            // V の初期化
            rep(i, maxdim+1) V.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            /// invV の初期化
            rep(i, maxdim+1) invV.push_back(unordered_map<simp_idx_t, unordered_set<simp_idx_t>>());
            // 2項係数の前計算
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
            // enclosing_radius の計算
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

        // 単体の最大辺の2端点を返す
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
            // row（heap）に coboundary を追加する
            // emg_opt: emerging pair による optimization を行うかどうか
            // encl_opt: enclosing radius による optimization を行うかどうか
            vector<pt_idx_t> vertex_list = get_simplex_vertices(dim, idx); // length: dim+1
            int k = dim;
            simp_idx_t term_1 = 0;
            simp_idx_t term_2 = 0;
            rep(i, dim+1) term_2 += binomial(vertex_list[dim-i], i+1);
            rep(i, N) {
                int j = N - i - 1;
                if (k >= 0 && j == vertex_list[dim-k]){
                    // term_2 から現在の k に対応する項を引く
                    if (k >= 0) term_2 -= binomial(vertex_list[dim-k], k+1);
                    else term_2 = 0;
                    // k を更新
                    k -= 1;
                    // term_1 に現在の k に対応する項を足す
                    term_1 += binomial(vertex_list[dim-k-1], k+3);
                    continue;
                }
                simp_idx_t coface_idx = term_1 + binomial(j, k+2) + term_2;
                diameter_t coface_diameter = get_diameter(dim+1, coface_idx);
                if (encl_opt && coface_diameter > enclosing_radius) continue;
                if (emg_opt && coface_diameter == get_diameter(dim, idx)) {
                    emg_opt = false;
                    if (death_to_birth[dim].count(coface_idx) == 0) return make_pair(true, coface_idx); 
                }
                if (push) row.push(DiameterEntry(dim+1, coface_idx, coface_diameter));
            }
            return make_pair(false, row.top().idx);
        }
        pair<bool, simp_idx_t> heappush_coboundary(dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt) {
            // row が指定されなかったときは push を行わない．
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
                if (encl_opt && face_diameter > enclosing_radius) continue;
                if (emg_opt && face_diameter == get_diameter(dim, idx)){
                    emg_opt = false;
                    if (birth_to_death[dim-1].count(face_idx) == 0) return make_pair(true, face_idx);
                }
                if (push) column.push(DiameterEntry(dim-1, face_idx, face_diameter));
            }
            return make_pair(false, column.top().idx);
        }
        pair<bool, simp_idx_t> heappush_boundary(dim_t dim, simp_idx_t idx, bool encl_opt, bool emg_opt) {
            // column が指定されなかったときは push を行わない
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

        void compute_ph(bool enclosing_opt=true, bool emgergent_opt=true, bool clearing_opt=true, bool get_inv=false) {
            // 各次元ごとに PH を調べる
            rep(_dim, 1, max(2U, maxdim+1)) {
                dim_t dim = _dim;
                // dim 次元の単体を列挙．clearing を行う場合は，対象となる単体を除外する．
                vector<DiameterEntry> simplex_list;
                rep(i, binomial(N, dim+1)){
                    simp_idx_t idx = i;
                    if (clearing_opt && death_to_birth[dim-1].count(idx)) continue; // 1つ前の次元の death は飛ばす（clearing）
                    // enclosing_radius を超えている場合はスキップ．この処理から，encl_opt を使う場合は W が完全には計算されない．
                    diameter_t diameter = get_diameter(dim, i);
                    if (enclosing_opt && (diameter > enclosing_radius)) continue;
                    // simplex_list に挿入
                    simplex_list.push_back(DiameterEntry(dim, idx, diameter));
                }
                // 各単体が apparent pair かどうかを調べる．そうでない単体だけ sorted_simplex_list に挿入する．
                vector<DiameterEntry> sorted_simplex_list;
                if (maxdim > 0) { // maxdim=0 のときはやるメリットがないし，やると death_to_birth の配列外参照になる
                    vector<unsigned int> is_apparent_list(simplex_list.size()); // ほんとは bool にしたいが，vector<bool> は良くないらしいので，int にする
                    vector<simp_idx_t> apparent_cofacet_list(simplex_list.size());
                    vector<thread> apparent_threads;
                    unsigned long long thread_loops = simplex_list.size() / num_threads + ((simplex_list.size() % num_threads) >= 1); // スレッドごとのループ数
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
                        // スレッドの情報を death_to_birth に集約
                        if (is_apparent_list[i]) {
                            simp_idx_t idx = simplex_list[i].idx;
                            death_to_birth[dim][apparent_cofacet_list[i]] = idx; 
                            W[dim][idx].insert(idx); invW[dim][idx].insert(idx);
                        }
                        else sorted_simplex_list.push_back(simplex_list[i]);
                    }
                    /// 以下，並列計算を用いない場合 ///
                    // rep(i, simplex_list.size()) { 
                    //     simp_idx_t idx = simplex_list[i].idx; 
                    //     check_zero_apparent(dim, idx, enclosing_opt, is_apparent_list[i], apparent_cofacet_list[i]);
                    //     if (is_apparent_list[i]) {
                    //         simp_idx_t idx = simplex_list[i].idx;
                    //         death_to_birth[dim][apparent_cofacet_list[i]] = idx; 
                    //         W[dim][idx].insert(idx); invW[dim][idx].insert(idx);
                    //     }
                    //     else sorted_simplex_list.push_back(simplex_list[i]);
                    // }
                    // cout << dim << " " << simplex_list.size() << " " << sorted_simplex_list.size() << endl;
                } else {
                    // simplex_list を sorted_simplex_list にコピー
                    sorted_simplex_list = create_copy(simplex_list);
                }
                // sorted_simplex_list を DiameterEntryGreaterDiameterSmallerIndex でソート = 登場順が遅いものから見ていく
                sort(sorted_simplex_list.begin(), sorted_simplex_list.end(), DiameterEntryGreaterDiameterSmallerIndex());
                if (dim == 1){
                    // DSU で 0-PH を計算
                    DSU dsu(N);
                    vector<DiameterEntry> _sorted_simplex_list; // 0-PH の death でなかったもの = 1-PH の birth になるものを格納
                    rep(_simp_entr, sorted_simplex_list.rbegin(), sorted_simplex_list.rend()){ // 注意：edge は iterator（ポインタ）要素を取り出す場合は *edge とする
                        vector<pt_idx_t> vertex_list = get_simplex_vertices(1, _simp_entr->idx);
                        pair<bool, pt_idx_t> dsu_ret = dsu.merge(vertex_list[0], vertex_list[1]);
                        if (dsu_ret.first) death_to_birth[0][_simp_entr->idx] = dsu_ret.second;
                        else _sorted_simplex_list.push_back(*_simp_entr);
                    }
                    sorted_simplex_list = _sorted_simplex_list;
                    reverse(sorted_simplex_list.begin(), sorted_simplex_list.end());
                }
                if (maxdim==0) break;
                // sorted_simplex_list の順に単体を見ていく
                rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){
                    W[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    invW[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    // row を _simp_entr の coboundary で初期化
                    Row row;
                    pair<bool, simp_idx_t> ret = heappush_coboundary(row, dim, _simp_entr->idx, enclosing_opt, emgergent_opt);
                    if (ret.first) {
                        death_to_birth[dim][ret.second] = _simp_entr->idx;
                        continue;
                    }
                    // row を，処理済みの行を使って掃き出し
                    while (clearing_opt || !row.empty()){ // clearing が false のときは，row が空になったときに終了する
                        DiameterEntry pivot_entry = row.top(); row.pop(); // dim+1 単体 の DiameterEntry．掃き出し中の行の一番左の非ゼロ要素．
                        if (!row.empty() && pivot_entry.idx == row.top().idx){
                            row.pop(); // F_2上で打ち消し合う
                        } else if (death_to_birth[dim].count(pivot_entry.idx)){
                            simp_idx_t target_simp = death_to_birth[dim][pivot_entry.idx]; // dim 単体の idx．この単体の W の行をつかって simp_idx を掃き出す．
                            symmetricDifference(W[dim][_simp_entr->idx], W[dim][target_simp]);
                            if (get_inv) invW[dim][target_simp].insert(_simp_entr->idx); // W と逆であることに注意．
                            rep(sweep_idx, W[dim][target_simp].begin(), W[dim][target_simp].end()) {
                                heappush_coboundary(row, dim, *sweep_idx, enclosing_opt, false);
                            }
                            // pivot_entry は row に戻す
                            row.push(pivot_entry);
                        } else {
                            death_to_birth[dim][pivot_entry.idx] = _simp_entr->idx;
                            break;
                        }
                    }
                }
            }
        }

        void compute_ph_right(bool enclosing_opt=true, bool emgergent_opt=true, bool get_inv=false) {
            // 各次元ごとに PH を調べる
            rep(_dim, maxdim+1) {
                dim_t dim = _dim;
                if (dim >= 1){
                    // dim 次元の単体を列挙．
                    vector<DiameterEntry> lower_simplex_list;
                    rep(i, binomial(N, dim+1)){
                        // [注意] 1つ前の次元で death となる単体は，絶対に掃き出されるので apparent にはならないから，含めてしまってもよい
                        simp_idx_t idx = i; 
                        // if (death_lower_simplex.count(idx)) continue;
                        // enclosing_radius を超えている場合はスキップ．
                        diameter_t diameter = get_diameter(dim, i);
                        if (enclosing_opt && (diameter > enclosing_radius)) continue;
                        // lower_simplex_list に挿入
                        lower_simplex_list.push_back(DiameterEntry(dim, idx, diameter));
                    }
                    // apparent pair を調べる．
                    vector<unsigned int> is_apparent_list(lower_simplex_list.size()); // ほんとは bool にしたいが，vector<bool> は良くないらしいので，int にする
                    vector<simp_idx_t> apparent_cofacet_list(lower_simplex_list.size());
                    vector<thread> apparent_threads;
                    unsigned long long thread_loops = lower_simplex_list.size() / num_threads + ((lower_simplex_list.size() % num_threads) >= 1); // スレッドごとのループ数
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
                        // スレッドの情報を birth_to_death に集約
                        if (is_apparent_list[i]) {
                            simp_idx_t idx = lower_simplex_list[i].idx;
                            simp_idx_t cofacet_idx = apparent_cofacet_list[i];
                            birth_to_death[dim][idx] = cofacet_idx;
                            V[dim][cofacet_idx].insert(cofacet_idx); invV[dim][cofacet_idx].insert(cofacet_idx);
                        }
                    }
                }
                // dim+1 次元の単体のうち，V[dim] の列に含まれていないものを列挙
                vector<DiameterEntry> sorted_simplex_list;
                rep(i, binomial(N, dim+2)){
                    simp_idx_t idx = i;
                    if (V[dim].count(idx)) continue;
                    // [TODO] compute_ph を先に計算している場合，(clearing_opt = true なら）death にならない単体は飛ばす
                    // enclosing_radius を超えている場合はスキップ．この処理から，encl_opt を使う場合は V が完全には計算されない．
                    diameter_t diameter = get_diameter(dim+1, i);
                    if (enclosing_opt && (diameter > enclosing_radius)) continue;
                    // sorted_simplex_list に挿入
                    sorted_simplex_list.push_back(DiameterEntry(dim+1, idx, diameter));
                }
                // dim+1 次元の単体を DiameterEntrySmallerDiameterGreaterIndex でソート＝登場順が早いものから見ていく
                sort(sorted_simplex_list.begin(), sorted_simplex_list.end(), DiameterEntrySmallerDiameterGreaterIndex());
                // dim=0 のときは，DSU で 0-PH を計算
                if (dim == 0){
                    DSU dsu(N);
                    rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){ // 注意：edge は iterator（ポインタ）要素を取り出す場合は *edge とする
                        vector<pt_idx_t> vertex_list = get_simplex_vertices(1, _simp_entr->idx);
                        pair<bool, pt_idx_t> dsu_ret = dsu.merge(vertex_list[0], vertex_list[1]);
                        if (dsu_ret.first) birth_to_death[0][dsu_ret.second] = _simp_entr->idx;
                    }
                    continue;
                }
                // sorted_simplex_list の順に，dim+1 単体を見ていく
                rep(_simp_entr, sorted_simplex_list.begin(), sorted_simplex_list.end()){
                    V[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    invV[dim][_simp_entr->idx].insert(_simp_entr->idx);
                    // column を _simp_entr の boundary で初期化
                    Column column;
                    pair<bool, simp_idx_t> ret = heappush_boundary(column, dim+1, _simp_entr->idx, enclosing_opt, emgergent_opt);
                    if (ret.first) {
                        birth_to_death[dim][ret.second] = _simp_entr->idx;
                        continue;
                    }
                    // row を，処理済みの行を使って掃き出し
                    while (!column.empty()){ // column: dim 単体の集まり．compute_ph とちがって，clearning をしていないので，column が 0 になることもある
                        DiameterEntry pivot_entry = column.top(); column.pop(); // dim 単体 の DiameterEntry．掃き出し中の列の一番下の非ゼロ要素．
                        if (!column.empty() && pivot_entry.idx == column.top().idx){
                            column.pop(); // F_2上で打ち消し合う
                        } else if (birth_to_death[dim].count(pivot_entry.idx)){
                            simp_idx_t target_simp = birth_to_death[dim][pivot_entry.idx]; // dim+1 単体の idx．この単体の V の列をつかって simp_idx を掃き出す．
                            symmetricDifference(V[dim][_simp_entr->idx], V[dim][target_simp]);
                            if (get_inv) invV[dim][target_simp].insert(_simp_entr->idx); // V と逆であることに注意．
                            rep(sweep_idx, V[dim][target_simp].begin(), V[dim][target_simp].end()) {
                                heappush_boundary(column, dim+1, *sweep_idx, enclosing_opt, false);
                            }
                            // pivot_entry は column に戻す
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