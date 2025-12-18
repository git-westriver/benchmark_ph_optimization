from ...ph_compute.ph_computation_library import RipsPH, Bar

import torch
import heapq
import itertools
from itertools import combinations

def vector_to_symmetric_matrix(
    d_vec: torch.Tensor,
    idx: torch.Tensor,
    num_pts: int,
) -> torch.Tensor:
    """
    上三角 (i<j) のベクトル d_vec から対称距離行列 D を作る。

    idx: shape (2, num_pairs) = torch.triu_indices(num_pts, num_pts, offset=1)
    """
    D = d_vec.new_zeros((num_pts, num_pts))
    i, j = idx[0], idx[1]
    D[i, j] = d_vec
    D[j, i] = d_vec  # 対称にコピー
    # 対角成分は 0 のまま（必要なら dist_mat.diag() をコピーしてもよい）
    return D

def dijkstra_over_swaps(
    dist_mat: torch.Tensor,
    n_strata: int,
    eps: float,
    one_swap_only: bool = True
) -> list[torch.Tensor]:
    """
    Parameters:
        dist_mat : (num_pts, num_pts) の距離行列 (torch.Tensor)
        n_strata : 見つけたいノード数
        eps : しきい値 (2*eps を超えるノードは無視)
        one_swap_only : 現在の距離行列から 1 swap で到達できるノードのみを探索．

    Returns: 
        list of torch.Tensor; each element represents the distance matrix of a stratum
    """
    assert dist_mat.dim() == 2 and dist_mat.size(0) == dist_mat.size(1)
    num_pts = dist_mat.size(0)
    device = dist_mat.device
    _dist_mat = dist_mat.detach().cpu()

    # 上三角 (対角除く) のインデックスを torch.triu_indices で取得
    # idx.shape == (2, num_pairs)
    idx = torch.triu_indices(num_pts, num_pts, offset=1, device=device)
    idx_i, idx_j = idx[0], idx[1]
    num_pairs = idx_i.numel()  # = num_pts * (num_pts - 1) // 2

    # 元の上三角距離ベクトル d0
    d0 = _dist_mat[idx_i, idx_j]  # shape: (num_pairs,)

    # スタートノード（元の行列）: perm = [0,1,2,...,num_pairs-1]
    start_perm = torch.arange(num_pairs, device=device, dtype=torch.long)

    def perm_to_cost(
        parent_perm: torch.Tensor, 
        parent_cost: float, 
        i: int, 
        j: int
    ) -> float:   
        a = parent_perm[i]
        b = parent_perm[j]

        vi = d0[i]
        vj = d0[j]
        va = d0[a]
        vb = d0[b]

        ci_old = (va - vi) ** 2
        cj_old = (vb - vj) ** 2
        ci_new = (vb - vi) ** 2
        cj_new = (va - vj) ** 2

        return parent_cost - ci_old - cj_old + ci_new + cj_new

    start_cost = 0. # perm_to_cost(start_perm)

    if one_swap_only:
        costs = (d0[idx_i] - d0[idx_j]) ** 2
        mask = costs <= eps

        masked_idx_i = idx_i[mask]
        masked_idx_j = idx_j[mask]
        masked_costs = costs[mask]

         # コストが小さい順に n_strata 件だけ取得
        # smallest n_strata -> largest of (-costs)
        k = min(n_strata, masked_costs.numel())
        vals, order = torch.topk(masked_costs, k=k, largest=False)
        i_sel = masked_idx_i[order]
        j_sel = masked_idx_j[order]

        # dist_mat が勾配をもつことが想定されるので，dist_mat を使って，permutation 後の距離行列を再構成
        ret = []
        for i, j in zip(i_sel, j_sel):
            d0 = dist_mat[idx_i, idx_j]  # shape: (num_pairs,), 勾配あり
            perm = start_perm.clone()
            perm[i], perm[j] = perm[j], perm[i]
            d_vec = d0[perm]  # shape: (num_pairs,)
            D = vector_to_symmetric_matrix(d_vec, idx, num_pts).to(device)
            ret.append(D)
        return ret

    else:
        # Dijkstra 風ヒープ（距離の小さい順に取り出す）
        heap = []
        counter = itertools.count()  # tie-breaker
        heapq.heappush(heap, (start_cost, next(counter), start_perm))

        visited = set()
        results = []

        while heap and len(results) < n_strata:
            cost, _, perm = heapq.heappop(heap)

            key = tuple(perm.tolist())
            if key in visited:
                continue
            visited.add(key)

            # しきい値チェック
            if cost <= 2.0 * eps:
                # 結果に追加（元の行列も cost=0 なのでここで入る）
                results.append({"cost": cost, "perm": perm.clone()})
            else:
                # このノード自体が 2*eps を超えるなら「無視」
                # そこから先の近傍も展開しない
                continue

            # 1 swap で到達できる近傍ノードをすべて生成
            # num_pairs が大きいと O(num_pairs^2) なので重い点に注意
            for i, j in combinations(range(num_pairs), 2):
                new_perm = perm.clone()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

                new_key = tuple(new_perm.tolist())
                if new_key in visited:
                    continue

                new_cost = perm_to_cost(perm, cost, i, j)
                if new_cost <= 2.0 * eps:
                    heapq.heappush(heap, (new_cost, next(counter), new_perm))

        # 念のため cost でソートして返す
        results.sort(key=lambda x: x["cost"])

        # dist_mat が勾配をもつことが想定されるので，dist_mat を使って，permutation 後の距離行列を再構成
        ret = []
        for result in results:
            d0 = dist_mat[idx_i, idx_j]  # shape: (num_pairs,), 勾配あり
            perm = result["perm"]
            d_vec = d0[perm]  # shape: (num_pairs,)
            D = vector_to_symmetric_matrix(d_vec, idx, num_pts).to(device)
            ret.append(D)
        return ret
