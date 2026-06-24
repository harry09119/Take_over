import torch
import math
from tqdm import tqdm

def re_prune(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: -1 이상의 int 텐서 (pad가 -1, 그 외는 유효값)
    k: -1로 만들고 싶은 유효값 개수

    반환: 일부 유효값이 -1로 바뀐 텐서 (x 복사본)
    """
    if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"x must be integer tensor, got {x.dtype}")

    # -1이 아닌 위치 (유효값 위치)
    mask_valid = (x != -1)
    num_valid = mask_valid.sum().item()

    if num_valid == 0:
        # 바꿀 게 없음
        return x.clone()

    if k > num_valid:
        # 원하는 개수가 전체 유효값보다 많으면, 일단 전체를 -1로 바꾸는 쪽으로 처리
        k = num_valid

    # 유효값들의 1D 인덱스
    valid_indices = mask_valid.nonzero(as_tuple=False)  # shape [num_valid, dim]
    # 이 중에서 k개를 랜덤하게 선택
    perm = torch.randperm(num_valid, device=x.device)
    chosen = valid_indices[perm[:k]]

    # 복사본 위에서 수정
    y = x.clone()
    # 다차원 인덱싱으로 선택된 위치들을 -1로 설정
    y[tuple(chosen.t())] = -1

    return y

def print_mat(matrix: torch.Tensor):
    """
    주어진 matrix에서 -1인 값을 "x"로 대체하여 출력합니다.
    모든 다른 값은 그대로 출력됩니다.
    """
    # 각 행을 순회하면서 -1은 "x"로, 그 외는 그대로 출력
    n,m = matrix.shape
    row_str = " ".join(f"{val:3d}" for val in range(0,m))
    print("\n",row_str)
    print("="*len(row_str))
    
    for i, row in enumerate(matrix.tolist()):
        row_str = " ".join(f"{'x':>3}" if val == -1 else f"{val:3d}" for val in row)
        row_str = row_str + " ||" + str(i)
        print(row_str)

def density_check(matrix):
    n,m = matrix.shape
    total = n * m
    if total == 0:
        return 0, 0, 0
    
    else:
        zero = 0
        for row in matrix.tolist():
            zero += row.count(-1)

        nonzero = total - zero
        density = round(nonzero/total, 3)
        #print("> Density: "+ f"{density:.2f}" + " with " + str(nonzero) + ", " + str(zero))
        return density, nonzero, m

def remove_empty(matrix: torch.Tensor) -> torch.Tensor:
    col_mask = ~(matrix == -1).all(dim=0)
    row_mask = ~(matrix == -1).all(dim=1)
    return matrix[row_mask][:, col_mask]

def remove_col(matrix):
    col_mask = ~(matrix == -1).all(dim=0)
    return matrix[:, col_mask]

def sparse_matrix(n,m,d, P):
    #matrix = torch.randn(n, m)
    matrix = torch.randperm(n*m).view(n, m)
    threshold = int(n * m * (1-d))
    if threshold > 0:
        matrix[matrix<threshold]=-1

    #mask = (torch.rand(n, m) < d).to(matrix.dtype)
    #pruned_matrix = matrix * mask

    #col_indices = torch.arange(m, dtype=torch.int32).unsqueeze(0).expand(n, m)
    #mask_int = mask.to(torch.int32)  # True -> 1, False -> 0
    #result = torch.where(mask_int.bool(), col_indices, torch.full_like(col_indices, -1))
    
    if(P):
        print("\n <<Generate Sparse Matrix with "+str(d)+"% Density>>")
        print_mat(matrix)
    
    return matrix

def eureka(matrix):
    n_rows, n_cols = matrix.shape
    if n_rows > 0 and n_cols > 0:
        aligned_matrix = torch.full((n_rows, n_cols), -1, dtype=matrix.dtype, device=matrix.device)

        for i, row in enumerate(matrix):
            nonzeros = row[row > -1]
            aligned_matrix[i, :len(nonzeros)] = nonzeros

        # 각 행의 -1보다 큰 값의 개수를 센다.
        nonzero_counts = (aligned_matrix > -1).sum(dim=1)
        avg_nonzero = math.ceil(nonzero_counts.sum().item() / n_rows)

        rebalanced_matrix = aligned_matrix.clone()

        # 아래 행부터 위로 올라가며 부족한 값을 위 행에서 가져온다.
        for row in range(n_rows - 1, 0, -1):
            count_row = nonzero_counts[row].item()
            if count_row < avg_nonzero:
                needed = avg_nonzero - count_row
                upper_row = row - 1

                available = nonzero_counts[upper_row].item()
                take = min(available, needed)

                if take > 0:
                    # 상위 행의 사용 가능한 값들 (앞쪽부터 available개)
                    nonzero_values = rebalanced_matrix[upper_row, :available]
                    # 현재 행에 상위 행의 값 중 뒤쪽 take개를 할당 (혹은 nonzero_values[:take]로 앞쪽 값을 가져올 수도 있음)
                    start = count_row
                    end = count_row + take
                    rebalanced_matrix[row, start:end] = nonzero_values[-take:]

                    # 상위 행에서 가져간 부분은 제거하고 -1로 채움
                    if take < available:
                        remaining = nonzero_values[:-take]
                    else:
                        remaining = torch.tensor([], dtype=matrix.dtype, device=matrix.device)
                    filler = torch.full((take,), -1, dtype=matrix.dtype, device=matrix.device)
                    new_upper = torch.cat((remaining, filler))
                    rebalanced_matrix[upper_row, :available] = new_upper

                    # nonzero_counts 업데이트 (텐서 요소이므로 + 연산 가능)
                    nonzero_counts[row] = nonzero_counts[row] + take
                    nonzero_counts[upper_row] = nonzero_counts[upper_row] - take

        # print_mat는 외부에서 정의된 함수라고 가정합니다.
        #print_mat(aligned_matrix)

        return remove_col(rebalanced_matrix)

    else:
        return None

def _count_valid_per_row(mat: torch.Tensor, pad_value=-1) -> torch.Tensor:
    """
    각 행마다 pad_value가 아닌 원소 개수를 센다.
    shape: [p, q] -> [p]
    """
    return (mat != pad_value).sum(dim=1)


def _compact_row(row: torch.Tensor, pad_value=-1) -> torch.Tensor:
    """
    한 행에서 pad_value가 아닌 값들을 왼쪽으로 몰고,
    나머지는 pad_value로 채운다.
    """
    vals = row[row != pad_value]
    out = torch.full_like(row, pad_value)
    out[:vals.numel()] = vals
    return out


def _displace_between_rows(
    mat: torch.Tensor,
    dst_row_idx: int,
    src_row_idx: int,
    n_disp: int,
    pad_value=-1,
):
    """
    행 src_row_idx 에서 dst_row_idx 로 n_disp개의 값을 "디스플레이스"한다.
    - src의 유효값들 중 tail n_disp개를 떼어내서
    - dst의 유효값 뒤에 append하는 방식.
    두 행 모두 다시 compact 형태로 재구성한다.

    반환값:
      mat_new, new_len_dst, new_len_src
    """
    if n_disp <= 0:
        # 아무 것도 옮기지 않는 경우
        return mat, None, None

    row_dst = mat[dst_row_idx]
    row_src = mat[src_row_idx]

    dst_vals = row_dst[row_dst != pad_value]
    src_vals = row_src[row_src != pad_value]

    assert n_disp <= src_vals.numel(), (
        f"Trying to displace {n_disp} elements, "
        f"but only {src_vals.numel()} available."
    )

    # src의 뒤에서 n_disp개 잘라서 dst로 붙인다.
    moved = src_vals[-n_disp:]          # tail
    new_src_vals = src_vals[:-n_disp]   # 남는 값들
    new_dst_vals = torch.cat([dst_vals, moved], dim=0)

    # 행 재구성 (compact + pad)
    new_dst_row = torch.full_like(row_dst, pad_value)
    new_src_row = torch.full_like(row_src, pad_value)
    new_dst_row[:new_dst_vals.numel()] = new_dst_vals
    new_src_row[:new_src_vals.numel()] = new_src_vals

    mat = mat.clone()
    mat[dst_row_idx] = new_dst_row
    mat[src_row_idx] = new_src_row

    return mat, new_dst_vals.numel(), new_src_vals.numel()


def eureka_decision(
    M: torch.Tensor,
    K: int,
    pad_value: int | float = -1,
):
    """
    EUREKA 논문의 Decision Problem (Algorithm 1)에 해당하는 부분.

    입력:
      - M: [p, q] 희소 행렬 (pad_value로 패딩된 형태도 허용)
      - K: 허용 가능한 최대 row length
      - pad_value: 패딩 값 (기본 -1)

    동작:
      - slack row 들( row length <= K )을 candidate base row로 잡아서
      - 각 base row에 대해, row i에서 rowabove=(i-1 mod p) 방향으로
        최대 p-1 단계까지 greedy하게 디스플레이스 시도.
      - 모든 행 길이가 <= K인 해를 찾으면 (O, True) 반환.
      - 어떤 base row로도 해가 없으면 (None, False) 반환.
    """
    p, q = M.shape
    if p == 0 or q == 0:
        return M.clone(), True

    # 먼저 모든 행을 compact 형태로 맞춰준다.
    mat0 = torch.stack([_compact_row(row, pad_value) for row in M], dim=0)
    lens0 = _count_valid_per_row(mat0, pad_value)

    # slack rows: length <= K
    slack_rows = torch.nonzero(lens0 <= K, as_tuple=False).flatten().tolist()
    if not slack_rows:
        return None, False

    # 각 slack row를 base row 후보로 시도
    for base_row in slack_rows:
        O = mat0.clone()
        lens = lens0.clone()

        row = base_row
        feasible = True

        # base row 포함 p개 row 중 base는 displace 하지 않으므로 최대 p-1단계
        for _ in range(p - 1):
            rowabove = (row - 1) % p

            C = int(lens[row].item())
            Cabove = int(lens[rowabove].item())
            slack = K - C  # 이 row가 추가로 수용 가능한 개수

            if slack <= 0 or Cabove == 0:
                # 가져올 필요/여유 없음 → 그냥 rowabove로 이동
                row = rowabove
                if lens[row] > K:
                    feasible = False
                    break
                continue

            n_disp = min(Cabove, slack)
            if n_disp > 0:
                O, new_len_row, new_len_above = _displace_between_rows(
                    O, row, rowabove, n_disp, pad_value
                )
                lens[row] = new_len_row
                lens[rowabove] = new_len_above

            row = rowabove

            # 새 row가 K를 초과하면 이 base row는 실패
            if lens[row] > K:
                feasible = False
                break

        # 모든 행이 <= K이면 K에 대한 해를 찾은 것
        if feasible and torch.all(lens <= K):
            return O, True

    # 어떤 base row로도 만족 못 함
    return None, False


def eureka_optimal(
    M: torch.Tensor,
    pad_value: int | float = -1,
):
    """
    EUREKA 논문의 full 알고리즘에 대응하는 PyTorch 구현.

    - Decision Problem을 만족하는 최소 K_opt를 찾고,
    - 그 때의 displaced 행렬 O_opt를 반환.

    입력:
      - M: [p, q] 희소 행렬 (pad_value로 패딩 가능)
      - pad_value: 패딩 값 (기본 -1)

    반환:
      - O_opt: 재배치된 행렬 (각 행 compact, 필요 시 오른쪽 all-pad 열 제거)
      - K_opt: 그 때의 최대 row length (critical path)

    해가 전혀 없는 경우 (이론상 거의 없지만), 
    fallback으로 compact만 한 M과 max row length를 반환.
    """
    p, q = M.shape
    if p == 0 or q == 0:
        return M.clone(), 0

    # 우선 compact
    M_comp = torch.stack([_compact_row(row, pad_value) for row in M], dim=0)
    lens = _count_valid_per_row(M_comp, pad_value)
    total = int(lens.sum().item())

    if total == 0:
        # 모두 pad_value면 K_opt = 0으로 본다
        return M_comp, 0

    # K 탐색 구간: [ceil(total/p), q]
    lower = math.ceil(total / p)
    upper = q

    best_O = None
    best_K = None

    lo, hi = lower, upper
    while lo <= hi:
        mid = (lo + hi) // 2
        O_mid, ok = eureka_decision(M_comp, mid, pad_value)
        if ok:
            # mid 길이로 가능한 해 존재 → 더 작은 K가 있는지 왼쪽 탐색
            best_O = O_mid
            best_K = mid
            hi = mid - 1
        else:
            # mid 길이로는 불가능 → 더 큰 K 필요
            lo = mid + 1

    if best_O is None:
        # 어떤 K에서도 해를 못 찾은 극단적인 경우 (거의 없음)
        return M_comp, int(lens.max().item())

    # 오른쪽 all-pad 열 제거 (q를 타이트하게 줄임)
    final_lens = _count_valid_per_row(best_O, pad_value)
    new_q = int(final_lens.max().item())
    if new_q < best_O.shape[1]:
        best_O = best_O[:, :new_q]

    return best_O, best_K


def column_combine_gpt(matrix, max_conflict, mux_size, P=False):
    """
    Column Combine (greedy) - 원래 동작을 보존하면서
    set union 생성/삭제를 줄여서 최적화한 버전.

    matrix: (n_rows, n_cols), 값 >=0 이면 nonzero, <0 이면 빈 슬롯으로 가정.
    max_conflict: 그룹 내에서 허용되는 누적 conflict 수.
    mux_size: 그룹당 수용 가능한 최대 column 개수.
    P: True면 결과 matrix를 print_mat으로 출력.
    """
    # 1) 비어 있는 행/열 제거 (원래 코드와 동일한 전처리)
    matrix = remove_empty(matrix)
    n_rows, n_cols = matrix.shape

    # 2) 각 열의 nonzero row index 집합 (>= 0인 위치)
    #    - set(torch.where(...)) 할 때 tolist() 결과를 바로 set으로.
    nonzero = []
    for ci in range(n_cols):
        rows = torch.where(matrix[:, ci] >= 0)[0]
        if rows.numel() == 0:
            nonzero.append(set())  # 빈 set
        else:
            nonzero.append(set(rows.tolist()))

    groups = []   # 각 그룹의 union된 row index 집합 (set)
    gidx   = []   # 각 그룹에 속한 열 인덱스 리스트 (list[int])
    gconf  = []   # 각 그룹의 누적 conflict 수 (list[int])

    # 3) 열 하나씩 greedy하게 그룹에 배치
    for ci in range(n_cols):
        col_set = nonzero[ci]
        if not col_set:
            # 해당 col에 nonzero가 전혀 없으면 skip (원래 코드와 동일)
            continue

        col_len = len(col_set)

        best_improve = 0    # improve > 0 인 그룹만 수용
        best_grp     = -1   # -1이면 새로운 그룹 생성
        best_conf    = 0    # 선택된 그룹에서 발생하는 conflict 수

        # 기존 그룹들 중에 넣을 수 있는 곳 탐색
        for gi, grp in enumerate(groups):
            # mux_size 넘으면 고려할 필요 없음 (조건 빠르게 탈락)
            if len(gidx[gi]) >= mux_size:
                continue

            # 교집합 크기를 먼저 계산
            # conflict = len(grp & col_set)
            common = grp & col_set
            conflict = len(common)

            # conflict 제약 위반 시 skip
            if conflict + gconf[gi] > max_conflict:
                continue

            # improve = |grp ∪ col_set| - |grp|
            #         = |col_set| - |grp ∩ col_set|
            #   → 실제 union set은 "후보로 선정된 그룹"에 대해서만 나중에 계산
            improve = col_len - conflict

            # 원래 코드처럼 "improve > best_improve"인 경우만 업데이트
            # (improve == 0 이면 새 그룹을 만드는 쪽으로 감)
            if improve > best_improve:
                best_improve = improve
                best_grp = gi
                best_conf = conflict

        # 4) 최종 배치 결정
        if best_grp < 0:
            # 수용 가능한 그룹이 없으면 새 그룹 생성
            groups.append(set(col_set))  # union은 col_set 그대로
            gidx.append([ci])
            gconf.append(0)
        else:
            # 선택된 그룹에 col 추가
            gidx[best_grp].append(ci)
            # 이 시점에서만 union 연산을 수행 (한 번만!)
            groups[best_grp] |= col_set
            gconf[best_grp] += best_conf

    # 5) packed matrix 만들기
    #    groups 수 = 행, 원래 row 수 = 열 (먼저 이렇게 만든 뒤 transpose)
    num_groups = len(groups)
    packed = torch.full((num_groups, n_rows), -1, dtype=torch.int32)

    # 각 group(=행)에 속한 모든 column의 nonzero 위치에 원래 column index 기입
    for gi, cols in enumerate(gidx):
        for ci in cols:
            for ri in nonzero[ci]:
                packed[gi, ri] = ci

    # 최종 packed는 (n_rows, num_groups)로 transpose
    packed = packed.transpose(0, 1).contiguous()

    # 그룹 길이 (각 그룹에 포함된 column 수)
    group_len = [len(lst) for lst in gidx]  # gidx는 빈 리스트가 없으므로 if lst 제거

    if P:
        print("\n <<Column Combine Sparse Matrix>>")
        print_mat(packed)

    #density_check(packed)

    return packed, group_len, gidx

def column_combine(matrix, max_conflict, mux_size, P=False):
    matrix = remove_empty(matrix)
    n_rows, n_cols = matrix.shape
    # 각 열의 nonzero row index 집합 (matrix >= 0)
    nonzero = [set(torch.where(matrix[:, ci] >= 0)[0].tolist()) for ci in range(n_cols)]
    
    groups = []       # 각 그룹의 union된 row index 집합
    gidx = []         # 각 그룹에 속한 열 인덱스 리스트
    gconf = []        # 각 그룹의 누적 conflict 수

    for ci in range(n_cols):
        if not nonzero[ci]:
            continue
        col_set = nonzero[ci]
        best_improve, best_grp, best_union, best_conflict = 0, -1, None, 0
        for gi, grp in enumerate(groups):
            union_set = grp | col_set
            improve = len(union_set) - len(grp)
            conflict = len(grp & col_set)
            if (conflict + gconf[gi] <= max_conflict) and (len(gidx[gi]) < mux_size):
                if improve > best_improve:
                    best_improve, best_grp, best_union, best_conflict = improve, gi, union_set, conflict
        if best_grp < 0:
            groups.append(set(col_set))
            gidx.append([ci])
            gconf.append(0)
        else:
            gidx[best_grp].append(ci)
            groups[best_grp] = best_union
            gconf[best_grp] += best_conflict

    packed = torch.full((len(groups), n_rows), -1, dtype=torch.int32)
    for gi, cols in enumerate(gidx):
        for ci in cols:
            for ri in nonzero[ci]:
                packed[gi, ri] = ci

    group_len = [len(lst) for lst in gidx if lst]
    
    packed = packed.transpose(0,1)

    if P:
        print("\n <<Column Combine Sparse Matrix>>")
        print_mat(packed)

    #density_check(packed)
        
    return packed, group_len, gidx

def column_combine_scrap(
    matrix,
    max_conflict,
    mux_size,
    P=False,
    max_regular_groups=None
):
    """
    열그룹 중심 Column Combining + 자투리 열그룹.

    하나의 실행 구간(window)은 다음과 같이 구성된다.

        Regular Group 0
        Regular Group 1
        ...
        Regular Group X-1
        Residual Group

    정규 열그룹에 원본 열을 결합할 때:
      - conflict가 없는 nonzero는 정규 열그룹에 배치
      - conflict가 발생한 nonzero는 자투리 열그룹에 배치

    자투리 열그룹으로 넘길 수 있는 조건:
      1. 정규 열그룹의 누적 conflict가 max_conflict 이하
      2. 자투리 열그룹 내부에서 row conflict가 없어야 함
      3. 자투리 열그룹이 사용하는 원본 열 수가 mux_size 이하
      4. 원본 열의 nonzero 중 최소 하나는 정규 열그룹에 배치되어야 함

    Args:
        matrix:
            입력 sparse matrix.
            matrix[ri, ci] >= 0인 위치를 nonzero로 간주한다.

        max_conflict:
            하나의 정규 열그룹에서 자투리 열그룹으로 넘길 수 있는
            누적 conflict nonzero 수.

        mux_size:
            하나의 정규 또는 자투리 열그룹이 사용할 수 있는
            최대 원본 열 인덱스 수.

        P:
            결과와 열그룹 정보를 출력할지 여부.

        max_regular_groups:
            하나의 자투리 열그룹을 공유하는 최대 정규 열그룹 수 X.
            자투리 activation의 최대 재사용 거리와 대응된다.

    Returns:
        packed:
            최종 packed matrix.
            shape = [n_rows, total_number_of_groups]

        group_len:
            각 열그룹이 사용하는 서로 다른 원본 열 수.

        gidx:
            각 열그룹이 사용하는 원본 열 인덱스 목록.

            각 실행 구간에서 자투리 열그룹은 정규 열그룹 뒤에 위치한다.
    """

    if mux_size <= 0:
        raise ValueError("mux_size must be greater than 0.")

    if max_regular_groups is None:
        max_regular_groups = mux_size

    if max_regular_groups <= 0:
        raise ValueError("max_regular_groups must be greater than 0.")

    if max_conflict < 0:
        raise ValueError("max_conflict must be greater than or equal to 0.")

    matrix = remove_empty(matrix)
    n_rows, n_cols = matrix.shape

    # 각 원본 열의 nonzero row index 집합
    nonzero = [
        set(torch.where(matrix[:, ci] >= 0)[0].tolist())
        for ci in range(n_cols)
    ]

    # 아직 어느 실행 구간에도 배치되지 않은 원본 열
    remaining = {
        ci for ci in range(n_cols)
        if nonzero[ci]
    }

    # 각 최종 packed 열의 {row: original_column_index}
    packed_columns = []

    # 각 최종 packed 열이 사용하는 원본 열 인덱스
    gidx = []

    # 출력 및 분석용 정보
    group_types = []
    group_conflicts = []
    group_windows = []

    window_id = 0

    while remaining:
        # ----------------------------------------------------------
        # 하나의 실행 구간에서 공유하는 자투리 열그룹
        # ----------------------------------------------------------
        residual_placement = {}
        residual_rows = set()
        residual_sources = set()

        regular_group_count = 0

        # ----------------------------------------------------------
        # 최대 X개의 정규 열그룹 생성
        # ----------------------------------------------------------
        while (
            remaining
            and regular_group_count < max_regular_groups
        ):
            # ------------------------------------------------------
            # 1. 정규 열그룹의 seed 선택
            # ------------------------------------------------------
            # nonzero가 많은 원본 열을 우선 선택한다.
            seed = max(
                remaining,
                key=lambda ci: (
                    len(nonzero[ci]),
                    -ci
                )
            )

            remaining.remove(seed)

            # 정규 열그룹에 실제 배치된 정보
            regular_placement = {
                ri: seed
                for ri in nonzero[seed]
            }

            regular_rows = set(nonzero[seed])
            regular_sources = [seed]
            regular_conflict = 0

            # ------------------------------------------------------
            # 2. 현재 정규 열그룹을 가능한 한 완성
            # ------------------------------------------------------
            while (
                remaining
                and len(regular_sources) < mux_size
            ):
                best_ci = None
                best_regular_rows = None
                best_residual_rows = None
                best_conflict = None
                best_score = None

                for ci in remaining:
                    col_rows = nonzero[ci]

                    # 정규 열그룹에 들어갈 nonzero
                    candidate_regular_rows = (
                        col_rows - regular_rows
                    )

                    # 정규 열그룹에서 conflict가 발생해
                    # 자투리 열그룹으로 넘어갈 nonzero
                    candidate_residual_rows = (
                        col_rows & regular_rows
                    )

                    conflict = len(candidate_residual_rows)
                    regular_gain = len(candidate_regular_rows)

                    # 원본 열의 모든 nonzero가 conflict라면,
                    # 해당 activation은 정규 열그룹에서 실제로
                    # 사용되지 않으므로 허용하지 않는다.
                    if regular_gain == 0:
                        continue

                    # 정규 열그룹의 누적 conflict 제한
                    if (
                        regular_conflict + conflict
                        > max_conflict
                    ):
                        continue

                    # conflict가 존재하면 자투리 열그룹 수용 가능성 검사
                    if conflict > 0:
                        # 자투리 열그룹 내부에서 같은 row를
                        # 두 번 사용할 수 없다.
                        if (
                            residual_rows
                            & candidate_residual_rows
                        ):
                            continue

                        # 자투리 열그룹에서 필요한 서로 다른
                        # activation 수는 mux_size 이하여야 한다.
                        new_residual_sources = (
                            residual_sources | {ci}
                        )

                        if (
                            len(new_residual_sources)
                            > mux_size
                        ):
                            continue

                    # --------------------------------------------------
                    # 후보 선택 점수
                    # --------------------------------------------------
                    # 1. 정규 열그룹에 많이 들어가는 열 우선
                    # 2. 자투리로 넘어가는 nonzero가 적은 열 우선
                    # 3. 전체 nonzero가 많은 열 우선
                    # 4. 원본 열 인덱스가 작은 열 우선
                    #
                    # 첫 항목은 실질적으로 얻는 이득에서
                    # residual 부담을 차감한 값이다.
                    score = (
                        regular_gain - conflict,
                        regular_gain,
                        -conflict,
                        len(col_rows),
                        -ci
                    )

                    if (
                        best_score is None
                        or score > best_score
                    ):
                        best_score = score
                        best_ci = ci
                        best_regular_rows = (
                            candidate_regular_rows
                        )
                        best_residual_rows = (
                            candidate_residual_rows
                        )
                        best_conflict = conflict

                # 더 이상 현재 정규 열그룹에 넣을 수 있는 열이 없음
                if best_ci is None:
                    break

                # --------------------------------------------------
                # 3. 선택된 원본 열을 정규/자투리 열그룹에 분할 배치
                # --------------------------------------------------
                for ri in best_regular_rows:
                    regular_placement[ri] = best_ci

                regular_rows.update(best_regular_rows)
                regular_sources.append(best_ci)

                if best_conflict > 0:
                    for ri in best_residual_rows:
                        residual_placement[ri] = best_ci

                    residual_rows.update(best_residual_rows)
                    residual_sources.add(best_ci)

                regular_conflict += best_conflict
                remaining.remove(best_ci)

            # ------------------------------------------------------
            # 4. 완성된 정규 열그룹 저장
            # ------------------------------------------------------
            packed_columns.append(regular_placement)
            gidx.append(list(regular_sources))

            group_types.append("regular")
            group_conflicts.append(regular_conflict)
            group_windows.append(window_id)

            regular_group_count += 1

        # ----------------------------------------------------------
        # 5. 해당 실행 구간의 자투리 열그룹 저장
        # ----------------------------------------------------------
        if residual_placement:
            packed_columns.append(residual_placement)
            gidx.append(sorted(residual_sources))

            group_types.append("residual")
            group_conflicts.append(0)
            group_windows.append(window_id)

        window_id += 1

    # --------------------------------------------------------------
    # 6. packed matrix 생성
    # --------------------------------------------------------------
    packed = torch.full(
        (n_rows, len(packed_columns)),
        -1,
        dtype=torch.int32,
        device=matrix.device
    )

    for gi, placement in enumerate(packed_columns):
        for ri, ci in placement.items():
            packed[ri, gi] = ci

    group_len = [
        len(source_columns)
        for source_columns in gidx
    ]

    # --------------------------------------------------------------
    # 7. lossless 검증
    # --------------------------------------------------------------
    expected_nonzeros = {
        (ri, ci)
        for ci in range(n_cols)
        for ri in nonzero[ci]
    }

    actual_nonzeros = {
        (ri, ci)
        for placement in packed_columns
        for ri, ci in placement.items()
    }

    if expected_nonzeros != actual_nonzeros:
        missing = expected_nonzeros - actual_nonzeros
        duplicated_or_invalid = actual_nonzeros - expected_nonzeros

        raise AssertionError(
            "Packed matrix is not lossless.\n"
            f"Missing nonzeros: {sorted(missing)[:20]}\n"
            f"Invalid nonzeros: {sorted(duplicated_or_invalid)[:20]}"
        )

    original_nnz = len(expected_nonzeros)
    packed_nnz = int((packed >= 0).sum().item())

    if original_nnz != packed_nnz:
        raise AssertionError(
            "Nonzero count mismatch: "
            f"original={original_nnz}, "
            f"packed={packed_nnz}"
        )

    # 각 열그룹 내부에서 MUX 제약 검사
    for gi, source_columns in enumerate(gidx):
        if len(set(source_columns)) > mux_size:
            raise AssertionError(
                f"Group {gi} exceeds mux_size: "
                f"{len(set(source_columns))} > {mux_size}"
            )

    if P:
        print("\n<<Column Combine with Residual Groups>>")
        print_mat(packed)

        print("\n<<Column Group Information>>")

        for gi, source_columns in enumerate(gidx):
            occupied_rows = len(packed_columns[gi])

            print(
                f"Group {gi}: "
                f"type={group_types[gi]}, "
                f"window={group_windows[gi]}, "
                f"columns={source_columns}, "
                f"group_len={len(source_columns)}, "
                f"conflicts={group_conflicts[gi]}, "
                f"occupied_rows={occupied_rows}"
            )

        regular_count = sum(
            group_type == "regular"
            for group_type in group_types
        )

        residual_count = sum(
            group_type == "residual"
            for group_type in group_types
        )

        print("\n<<Summary>>")
        print(f"Original columns : {n_cols}")
        print(f"Regular groups   : {regular_count}")
        print(f"Residual groups  : {residual_count}")
        print(f"Total groups     : {len(packed_columns)}")
        print(f"Original NNZ     : {original_nnz}")
        print(f"Packed NNZ       : {packed_nnz}")

    return packed, group_len, gidx

def pad_id(gidx, mux_size):
    padded = []
    for cols in gidx:
        cols_pad = list(cols)
        
        while len(cols_pad) < mux_size:
            cols_pad.append(-1)
        
        padded.append(cols_pad)
    return padded


def reorder_tensor(A, mod = "d"):
    row_counts = (A > -1).sum(dim=1)
    if mod == "a":
        order = False
    else:
        order = True
    row_order = torch.argsort(row_counts, descending=order)

    #col_counts = (A > -1).sum(dim=0)
    #col_order = torch.argsort(col_counts, descending=True)

    reordered_A = A[row_order]#[:, col_order]
    return reordered_A

def opf(now_t, now_col_in_grps, next_t, mux_size, sa_size = 16, ope = True):
    now_t_l = reorder_tensor(now_t,"d")
    #now_t_l, _, now_col_in_grps = column_combine(now_t_l, 0, mux_size, False)

    next_t_l = reorder_tensor(next_t, "a")
    next_col_descend = torch.argsort((next_t_l > -1).sum(dim=0), descending=True).tolist()
    
    _,now_nz,_=density_check(now_t)
    _,next_nz,_=density_check(next_t)
    nonzeros = now_nz + next_nz
    if ope:
        out_of_window = 0
    else:
        out_of_window = 2*sa_size - 1
    for i, col in enumerate(now_t_l.transpose(0,1)):
        if i >= out_of_window:
            next_col_able = now_col_in_grps[i]
            next_col_able = sorted(next_col_able, key=next_col_descend.index)
            for j, val in enumerate(col):
                if val == -1:
                    able_cols = [(next_col_able[i],x) for i, x in enumerate(next_t_l[j, next_col_able]) if x > -1]
                    if able_cols:
                        index, nonzero = able_cols[0]
                        now_t_l[j,i] = nonzero
                        next_t_l[j,index] = -1
                        #print("Move: ",(j,i)," <- ",(j,int(able_cols[index])),int(nonzero))
        
    _,n_now_nz,_=density_check(now_t_l)
    _,n_next_nz,_=density_check(next_t_l)
    
    if nonzeros != (n_now_nz + n_next_nz):
        print(">>>>Row Scatter Error. Please Edit Code",)
        print(">> "+str(now_nz)+"->"+str(n_now_nz)+" = "+str(n_now_nz-now_nz))
        print(">> "+str(next_nz)+"->"+str(n_next_nz)+" = "+str(next_nz-n_next_nz))       

    return now_t_l, next_t_l

def row_grouping(A: torch.Tensor, B: int, C: int, P: bool=False):
    groups = []  # 각 그룹의 nonzero column index의 union (set)
    gidx = []    # 각 그룹에 속한 row index 리스트

    n_rows, n_cols = A.shape

    for r in range(n_rows):
        # 현재 row의 nonzero column index 집합 (0이 아닌 값 기준)
        r_nonzero = set(torch.nonzero(A[r] >= 0).view(-1).tolist())
        # torch.nonzero의 결과가 단일 정수인 경우도 대비
        if isinstance(r_nonzero, int):
            r_nonzero = {r_nonzero}

        candidate = None
        best_overlap = -1

        # 이미 생성된 그룹들 중 아직 row를 추가할 수 있는(아직 C개 미만인) 그룹을 탐색합니다.
        for i, group in enumerate(groups):
            if len(gidx[i]) < C:
                overlap = len(group.intersection(r_nonzero))
                if overlap > best_overlap:
                    best_overlap = overlap
                    candidate = i

        # 기존 그룹 중 어느 곳에도 overlap이 없거나 후보 그룹이 없는 경우
        if candidate is None:
            # 만약 아직 그룹 수가 B 미만이면 새 그룹을 생성
            if len(groups) < B:
                groups.append(set(r_nonzero))
                gidx.append([r])
            else:
                # 이미 B개의 그룹이 모두 생성되어 있다면 이 row는 배정하지 않습니다.
                continue
        else:
            # 만약 후보 그룹의 overlap이 0이고, 새로운 그룹을 생성할 수 있다면
            # 서로 다른 nonzero 패턴을 갖도록 새 그룹 생성하는 것이 유리할 수 있음.
            if best_overlap == 0 and len(groups) < B:
                groups.append(set(r_nonzero))
                gidx.append([r])
            else:
                # 후보 그룹에 할당
                gidx[candidate].append(r)
                groups[candidate].update(r_nonzero)

    # 각 그룹별 row index를 packed 텐서로 정리합니다.
    num_groups = len(gidx)
    packed = torch.full((num_groups, C), -1, dtype=torch.int32)
    for i, rows in enumerate(gidx):
        for j, row_idx in enumerate(rows):
            if j < C:
                packed[i, j] = row_idx

    if P:
        print("Packed groups:")
        print(packed)

    group_lengths = [len(x) for x in gidx]
    return packed

"""
if __name__ == '__main__':
    n, m, s, r, P= 256, 256, 16, 1, False
    ratio = 1
    d_list = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    conflict, mux_size = 0.25, 8

    for d in d_list:
        n_r = int(n*ratio)
        m_r = int(n/ratio)
        compare = [0,0,0,0,0,0,0,0]
        for i in tqdm(range(r)):
            total_t = sparse_matrix(n_r,m_r,d,False)

            density, nonzero, length = density_check(total_t)
            origin_nz = nonzero
            origin_len = length

            total_t = remove_empty(total_t)
           
            n_new, m_new = total_t.shape
            n_tiles = (n_new + s - 1)//s

            # Column Combine Test Start
            
            first_len = 0
            first_nz = 0

            analyze_t = []
            
            tile_wise = False

            if tile_wise: 
                for i in range(n_tiles):
                    start = i*s
                    end = (i+1)*s if (i+1)*s <= m_new else m_new
                    part_t = total_t[start:end]
                    colcom_p_t, _, col_info = column_combine(part_t, conflict, mux_size)

                    density, nonzero, length = density_check(colcom_p_t) 
                    first_len += length
                    first_nz += nonzero
            
            else:
                colcom_t, _, _ = column_combine(total_t, n_new*conflict, mux_size)
                density, nonzero, length = density_check(colcom_t) 
                first_len += length
                first_nz += nonzero

            # Column Combine Test End

            # Eureka Test Start
            mux_size_ = mux_size * 2
            third_len = 0
            third_nz = 0

            
            for i in range(n_tiles):
                start = i*s
                end = min((i+1)*s, n_new)
                part_t = total_t[start:end]
                part_t = reorder_tensor(part_t, "d")

                part_t_nums = (m_new + mux_size_ - 1) // mux_size_
                for j in range(part_t_nums):
                    part_part_t = part_t[:,j*mux_size_:(j+1)*mux_size_]
                    eureka_t = eureka(part_part_t)

                    density, nonzero, length = density_check(eureka_t)
                    third_len += length
                    third_nz += nonzero
            
            if third_nz != origin_nz:
                print(">> Error: Eureka code broke!!!")
            # Eureka Test End

            # Cross Combine Test Start
                  
            result_t = []
            analyze_2_t = []

            second_cycle = 0
            second_len = 0
            second_nz = 0

            total_t = reorder_tensor(total_t, "a")

            for i in range(n_tiles-1):
                now_start = i*s
                now_end = (i+1)*s
                now_t = total_t[now_start:now_end]

                next_start = now_end
                next_end = (i+2)*s if (i+2)*s <= n_new else n_new
                next_t = total_t[next_start:next_end]

                diff = s - (next_end-next_start)
                if diff > 0:
                    next_t = torch.cat([next_t, torch.full((diff, next_t.size(1)), -1, dtype=next_t.dtype)], 0)

                crocom_t, pruned_t = ctf(now_t, next_t, mux_size)
                pruned_t = pruned_t if diff == 0 else pruned_t[:-diff]
                total_t[next_start:next_end].copy_(pruned_t)
                
                density, nonzero, length = density_check(crocom_t)
                #result_t.append(combined_t)
                #analyze_2_t.append(density)
                
                #second_cycle += length
                second_len += length
                second_nz += nonzero

                if i == (n_tiles-2):
                    crocom_t, _, _ = column_combine(pruned_t, 0, mux_size)
                    density, nonzero, length = density_check(crocom_t)
                    #result_t.append(combined_t)
                    #analyze_2_t.append(density)
                    #second_cycle += length
                    second_len += length
                    second_nz += nonzero

            if second_nz != origin_nz:
                print(">> Error: Cross Combine code broke!!!")

            # Cross Combine Test End
            compare[0] += origin_len * n_r / s
            compare[1] += origin_nz / (n_r * m_r) * 100
            compare[2] += first_len * n_tiles if not tile_wise else first_len
            compare[3] += first_nz / (n_new * first_len) * 100 if not tile_wise else first_nz / (s*first_len) * 100
            compare[4] += second_len
            compare[5] += second_nz / (s * second_len) * 100
            compare[6] += third_len
            compare[7] += third_nz / (s * third_len) * 100

        compare = [round(x/r,2) for x in compare]
        
        print(">> [Test Result in shape("+str(d*100)+"%)]<<")
        for i in range(len(compare)//2):
            look = compare[i*2:(i+1)*2]
            print(">> "+str(i)+": ",look)
            if i == 1:
                pruned = round((origin_nz - first_nz)/(n_r*m_r)*100,2)
                pruned_r = round((origin_nz-first_nz)/origin_nz*100,2)
                print(">> pruned: ",pruned,"%(total), ",pruned_r,"%(left)")
        #print(">> Result in ["+str(n_r)+" X "+str(m_r)+"]: "+" ".join(f"{x:0.2f}" for x in compare))
        """

# =============================================================================
# Current lossless Column Combining for Overlap SA
# - Cg % parallel_groups lane-local residual sharing
# - multiple residual groups per lane
# - source column indices are preserved relative to the input tile
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _iter_mask_rows(mask: int) -> Iterable[int]:
    while mask:
        lowest_bit = mask & -mask
        yield lowest_bit.bit_length() - 1
        mask ^= lowest_bit


@dataclass(frozen=True)
class PreparedColumnsCurrent:
    n_rows: int
    n_cols: int
    column_masks: List[int]
    column_nnz: List[int]
    ranked_columns: List[int]
    remaining: Set[int]
    device: torch.device


def prepare_columns_current(matrix: torch.Tensor) -> PreparedColumnsCurrent:
    """
    Prepare an S x B tile without compacting its columns.

    Keeping the original input-tile column indices is essential because CTF
    uses gidx to select the corresponding activation from the next row tile.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    n_rows, n_cols = matrix.shape
    column_masks: List[int] = []
    column_nnz: List[int] = []

    for ci in range(n_cols):
        rows = torch.where(matrix[:, ci] >= 0)[0].tolist()
        mask = 0
        for ri in rows:
            mask |= 1 << int(ri)
        column_masks.append(mask)
        column_nnz.append(len(rows))

    ranked_columns = sorted(
        (ci for ci, count in enumerate(column_nnz) if count > 0),
        key=lambda ci: (-column_nnz[ci], ci),
    )
    remaining = set(ranked_columns)

    return PreparedColumnsCurrent(
        n_rows=n_rows,
        n_cols=n_cols,
        column_masks=column_masks,
        column_nnz=column_nnz,
        ranked_columns=ranked_columns,
        remaining=remaining,
        device=matrix.device,
    )


def _select_seed_current(remaining: Set[int], ranked_columns: Sequence[int]) -> int:
    for ci in ranked_columns:
        if ci in remaining:
            return ci
    raise RuntimeError("No seed exists although remaining is nonempty")


def _candidate_columns_current(
    remaining: Set[int],
    ranked_columns: Sequence[int],
    search_limit: Optional[int],
) -> Iterable[int]:
    count = 0
    for ci in ranked_columns:
        if ci not in remaining:
            continue
        yield ci
        count += 1
        if search_limit is not None and count >= search_limit:
            break


@dataclass
class ResidualGroupStateCurrent:
    mask: int = 0
    sources: Set[int] = field(default_factory=set)
    placement: Dict[int, int] = field(default_factory=dict)


@dataclass
class PackingMetadataCurrent:
    group_types: List[str] = field(default_factory=list)
    group_blocks: List[int] = field(default_factory=list)
    group_lanes: List[int] = field(default_factory=list)
    group_conflicts: List[int] = field(default_factory=list)

    regular_groups: int = 0
    residual_groups: int = 0
    padding_groups: int = 0
    regular_cycles: int = 0
    residual_cycles: int = 0
    total_cycles: int = 0
    blocks: int = 0
    residual_nnz: int = 0
    residual_source_references: int = 0


@dataclass
class PackingResultCurrent:
    scheduled_packed: torch.Tensor
    group_len: List[int]
    gidx: List[List[int]]
    metadata: PackingMetadataCurrent

    @property
    def packed(self) -> torch.Tensor:
        return self.scheduled_packed


def _build_packed_current(
    n_rows: int,
    placements: Sequence[Dict[int, int]],
    device: torch.device,
) -> torch.Tensor:
    packed = torch.full(
        (n_rows, len(placements)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for gi, placement in enumerate(placements):
        for ri, ci in placement.items():
            packed[ri, gi] = ci
    return packed


def _validate_lossless_current(
    prepared: PreparedColumnsCurrent,
    placements: Sequence[Dict[int, int]],
    gidx: Sequence[Sequence[int]],
    mux_size: int,
) -> None:
    expected = {
        (ri, ci)
        for ci, mask in enumerate(prepared.column_masks)
        for ri in _iter_mask_rows(mask)
    }
    actual_list = [
        (ri, ci)
        for placement in placements
        for ri, ci in placement.items()
    ]
    actual = set(actual_list)

    if expected != actual:
        raise AssertionError(
            "Current Column Combining is not lossless.\n"
            f"Missing: {sorted(expected - actual)[:20]}\n"
            f"Invalid: {sorted(actual - expected)[:20]}"
        )
    if len(actual_list) != len(actual):
        raise AssertionError("A nonzero was duplicated across packed groups")

    for gi, sources in enumerate(gidx):
        if len(set(sources)) > mux_size:
            raise AssertionError(
                f"Group {gi} exceeds mux_size: {len(set(sources))} > {mux_size}"
            )


def _append_scheduled_current(
    placements: List[Dict[int, int]],
    gidx: List[List[int]],
    metadata: PackingMetadataCurrent,
    *,
    placement: Dict[int, int],
    sources: Sequence[int],
    group_type: str,
    block_id: int,
    lane: int,
    conflicts: int,
    parallel_groups: int,
) -> None:
    physical_cg = len(placements)
    if physical_cg % parallel_groups != lane:
        raise AssertionError(
            f"Lane alignment error: Cg={physical_cg}, "
            f"Cg%P={physical_cg % parallel_groups}, expected={lane}"
        )

    placements.append(dict(placement))
    gidx.append(list(sources))
    metadata.group_types.append(group_type)
    metadata.group_blocks.append(block_id)
    metadata.group_lanes.append(lane)
    metadata.group_conflicts.append(conflicts)

    if group_type.startswith("regular") and "padding" not in group_type:
        metadata.regular_groups += 1
    elif group_type.startswith("residual") and "padding" not in group_type:
        metadata.residual_groups += 1
    elif "padding" in group_type:
        metadata.padding_groups += 1


def column_combine_lossless_current(
    matrix: torch.Tensor,
    mux_size: int,
    P: bool = False,
    candidate_search_limit: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], List[List[int]]]:
    """Strict tile-wise lossless CC using the same group-centred search."""
    if mux_size <= 0:
        raise ValueError("mux_size must be greater than 0")

    prepared = prepare_columns_current(matrix)
    remaining = set(prepared.remaining)
    placements: List[Dict[int, int]] = []
    gidx: List[List[int]] = []

    while remaining:
        seed = _select_seed_current(remaining, prepared.ranked_columns)
        remaining.remove(seed)

        group_mask = prepared.column_masks[seed]
        sources = [seed]
        placement = {ri: seed for ri in _iter_mask_rows(group_mask)}

        while remaining and len(sources) < mux_size:
            best_ci: Optional[int] = None
            best_score: Optional[Tuple[int, int]] = None

            for ci in _candidate_columns_current(
                remaining, prepared.ranked_columns, candidate_search_limit
            ):
                mask = prepared.column_masks[ci]
                if group_mask & mask:
                    continue
                score = (prepared.column_nnz[ci], -ci)
                if best_score is None or score > best_score:
                    best_score = score
                    best_ci = ci

            if best_ci is None:
                break

            mask = prepared.column_masks[best_ci]
            for ri in _iter_mask_rows(mask):
                placement[ri] = best_ci
            group_mask |= mask
            sources.append(best_ci)
            remaining.remove(best_ci)

        placements.append(placement)
        gidx.append(sources)

    _validate_lossless_current(prepared, placements, gidx, mux_size)
    packed = _build_packed_current(prepared.n_rows, placements, prepared.device)
    if P:
        print("\n<<Current Strict Lossless Column Combining>>")
        print_mat(packed)
    return packed, [len(x) for x in gidx], gidx


def column_combine_modulo_residual_current(
    matrix: torch.Tensor,
    mux_size: int,
    reuse_depth: int,
    max_residual_groups_per_lane: int,
    parallel_groups: int = 4,
    max_conflict: Optional[int] = None,
    new_residual_group_penalty: int = 1,
    P: bool = False,
    candidate_search_limit: Optional[int] = None,
) -> PackingResultCurrent:
    """
    Current lossless CC used by CTC.

    Groups with the same Cg % parallel_groups share lane-local residual pools.
    Residual groups are emitted after each regular block, with padding slots to
    preserve the physical lane index. Both regular and residual groups may be
    used later by cross_tile_fill_all_groups_current().
    """
    if mux_size <= 0:
        raise ValueError("mux_size must be greater than 0")
    if reuse_depth <= 0:
        raise ValueError("reuse_depth must be greater than 0")
    if max_residual_groups_per_lane <= 0:
        raise ValueError("max_residual_groups_per_lane must be greater than 0")
    if parallel_groups <= 0:
        raise ValueError("parallel_groups must be greater than 0")
    if max_conflict is not None and max_conflict < 0:
        raise ValueError("max_conflict must be nonnegative or None")

    prepared = prepare_columns_current(matrix)
    remaining = set(prepared.remaining)
    placements: List[Dict[int, int]] = []
    gidx: List[List[int]] = []
    metadata = PackingMetadataCurrent()

    block_capacity = parallel_groups * reuse_depth
    block_id = 0

    while remaining:
        lane_residual_pools: List[List[ResidualGroupStateCurrent]] = [
            [] for _ in range(parallel_groups)
        ]
        regular_placements: List[Dict[int, int]] = []
        regular_sources_all: List[List[int]] = []
        regular_conflicts_all: List[int] = []
        regular_slots_used = 0

        while remaining and regular_slots_used < block_capacity:
            lane = regular_slots_used % parallel_groups
            residual_pool = lane_residual_pools[lane]

            seed = _select_seed_current(remaining, prepared.ranked_columns)
            remaining.remove(seed)

            regular_mask = prepared.column_masks[seed]
            regular_sources = [seed]
            regular_placement = {
                ri: seed for ri in _iter_mask_rows(regular_mask)
            }
            regular_conflict_count = 0

            while remaining and len(regular_sources) < mux_size:
                best_ci: Optional[int] = None
                best_accepted_mask = 0
                best_conflict_mask = 0
                best_conflict_count = 0
                best_residual_index: Optional[int] = None
                best_opens_residual = False
                best_score = None

                for ci in _candidate_columns_current(
                    remaining,
                    prepared.ranked_columns,
                    candidate_search_limit,
                ):
                    column_mask = prepared.column_masks[ci]
                    conflict_mask = regular_mask & column_mask
                    accepted_mask = column_mask & ~regular_mask
                    conflict_count = conflict_mask.bit_count()
                    accepted_count = accepted_mask.bit_count()

                    # The activation must be consumed in the regular group first.
                    if accepted_count == 0:
                        continue
                    if (
                        max_conflict is not None
                        and regular_conflict_count + conflict_count > max_conflict
                    ):
                        continue

                    residual_index: Optional[int] = None
                    opens_residual = False
                    residual_fill = 0
                    residual_source_fill = 0

                    if conflict_count > 0:
                        best_fit = None
                        for ri, residual_group in enumerate(residual_pool):
                            if residual_group.mask & conflict_mask:
                                continue
                            if len(residual_group.sources | {ci}) > mux_size:
                                continue
                            fit = (
                                residual_group.mask.bit_count(),
                                len(residual_group.sources),
                                -ri,
                            )
                            if best_fit is None or fit > best_fit:
                                best_fit = fit
                                residual_index = ri

                        if residual_index is None:
                            if len(residual_pool) >= max_residual_groups_per_lane:
                                continue
                            residual_index = len(residual_pool)
                            opens_residual = True
                        else:
                            selected = residual_pool[residual_index]
                            residual_fill = selected.mask.bit_count()
                            residual_source_fill = len(selected.sources)

                    net_gain = (
                        accepted_count
                        - conflict_count
                        - new_residual_group_penalty * int(opens_residual)
                    )
                    score = (
                        net_gain,
                        -int(opens_residual),
                        accepted_count,
                        -conflict_count,
                        residual_fill,
                        residual_source_fill,
                        prepared.column_nnz[ci],
                        -ci,
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_ci = ci
                        best_accepted_mask = accepted_mask
                        best_conflict_mask = conflict_mask
                        best_conflict_count = conflict_count
                        best_residual_index = residual_index
                        best_opens_residual = opens_residual

                if best_ci is None:
                    break

                for row in _iter_mask_rows(best_accepted_mask):
                    regular_placement[row] = best_ci
                regular_mask |= best_accepted_mask
                regular_sources.append(best_ci)

                if best_conflict_count > 0:
                    if best_residual_index is None:
                        raise AssertionError("Residual destination is missing")
                    if best_opens_residual:
                        if best_residual_index != len(residual_pool):
                            raise AssertionError("New residual index mismatch")
                        residual_pool.append(ResidualGroupStateCurrent())
                    destination = residual_pool[best_residual_index]
                    if destination.mask & best_conflict_mask:
                        raise AssertionError("Unexpected residual row conflict")
                    for row in _iter_mask_rows(best_conflict_mask):
                        destination.placement[row] = best_ci
                    destination.mask |= best_conflict_mask
                    destination.sources.add(best_ci)

                regular_conflict_count += best_conflict_count
                remaining.remove(best_ci)

            regular_placements.append(regular_placement)
            regular_sources_all.append(regular_sources)
            regular_conflicts_all.append(regular_conflict_count)
            regular_slots_used += 1

        regular_slots_scheduled = (
            math.ceil(regular_slots_used / parallel_groups) * parallel_groups
        )
        for slot in range(regular_slots_scheduled):
            lane = slot % parallel_groups
            if slot < regular_slots_used:
                _append_scheduled_current(
                    placements,
                    gidx,
                    metadata,
                    placement=regular_placements[slot],
                    sources=regular_sources_all[slot],
                    group_type=f"regular_lane_{lane}",
                    block_id=block_id,
                    lane=lane,
                    conflicts=regular_conflicts_all[slot],
                    parallel_groups=parallel_groups,
                )
            else:
                _append_scheduled_current(
                    placements,
                    gidx,
                    metadata,
                    placement={},
                    sources=[],
                    group_type=f"regular_padding_lane_{lane}",
                    block_id=block_id,
                    lane=lane,
                    conflicts=0,
                    parallel_groups=parallel_groups,
                )

        regular_cycles = regular_slots_scheduled // parallel_groups
        metadata.regular_cycles += regular_cycles

        residual_rounds = max(
            (len(pool) for pool in lane_residual_pools), default=0
        )
        for residual_round in range(residual_rounds):
            for lane in range(parallel_groups):
                pool = lane_residual_pools[lane]
                if residual_round < len(pool):
                    group = pool[residual_round]
                    _append_scheduled_current(
                        placements,
                        gidx,
                        metadata,
                        placement=group.placement,
                        sources=sorted(group.sources),
                        group_type=f"residual_round_{residual_round}_lane_{lane}",
                        block_id=block_id,
                        lane=lane,
                        conflicts=0,
                        parallel_groups=parallel_groups,
                    )
                    metadata.residual_nnz += len(group.placement)
                    metadata.residual_source_references += len(group.sources)
                else:
                    _append_scheduled_current(
                        placements,
                        gidx,
                        metadata,
                        placement={},
                        sources=[],
                        group_type=f"residual_padding_round_{residual_round}_lane_{lane}",
                        block_id=block_id,
                        lane=lane,
                        conflicts=0,
                        parallel_groups=parallel_groups,
                    )

        metadata.residual_cycles += residual_rounds
        metadata.total_cycles += regular_cycles + residual_rounds
        metadata.blocks += 1
        block_id += 1

    _validate_lossless_current(prepared, placements, gidx, mux_size)
    packed = _build_packed_current(prepared.n_rows, placements, prepared.device)

    if packed.shape[1] % parallel_groups != 0:
        raise AssertionError("Scheduled packed width is not lane aligned")
    if metadata.total_cycles * parallel_groups != packed.shape[1]:
        raise AssertionError("Cycle count and packed width disagree")

    if P:
        print("\n<<Current Modulo Residual Column Combining>>")
        print_mat(packed)
        print(
            f"cycles={metadata.total_cycles}, regular={metadata.regular_cycles}, "
            f"residual={metadata.residual_cycles}, padding={metadata.padding_groups}"
        )

    return PackingResultCurrent(
        scheduled_packed=packed,
        group_len=[len(x) for x in gidx],
        gidx=gidx,
        metadata=metadata,
    )


def cross_tile_fill_all_groups_current(
    now_packed: torch.Tensor,
    now_group_sources: Sequence[Sequence[int]],
    next_tile: torch.Tensor,
    *,
    pad_value: int = -1,
    reorder_rows: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply CTF to every non-padding packed group.

    Regular and residual groups are both eligible. Padding groups naturally
    have an empty source list and are skipped. A next-tile nonzero can move to
    a hole only when its source column already belongs to that packed group,
    so no additional MUX input is introduced.
    """
    if now_packed.ndim != 2 or next_tile.ndim != 2:
        raise ValueError("now_packed and next_tile must be two-dimensional")
    if now_packed.shape[0] != next_tile.shape[0]:
        raise ValueError(
            f"row mismatch: now={now_packed.shape[0]}, next={next_tile.shape[0]}"
        )
    if now_packed.shape[1] != len(now_group_sources):
        raise ValueError("now_group_sources length must match packed width")

    now = now_packed.clone()
    nxt = next_tile.clone()
    if reorder_rows:
        now = reorder_tensor(now, "d")
        nxt = reorder_tensor(nxt, "a")

    before = int((now != pad_value).sum().item()) + int(
        (nxt != pad_value).sum().item()
    )

    next_counts = (nxt != pad_value).sum(dim=0)
    source_order = torch.argsort(next_counts, descending=True).tolist()
    source_rank = {source: rank for rank, source in enumerate(source_order)}

    moved = 0
    n_source_cols = nxt.shape[1]

    for group_index in range(now.shape[1]):
        sources = sorted(
            {
                int(ci)
                for ci in now_group_sources[group_index]
                if 0 <= int(ci) < n_source_cols
            },
            key=lambda ci: source_rank.get(ci, n_source_cols),
        )
        if not sources:
            continue

        for row_index in range(now.shape[0]):
            if int(now[row_index, group_index].item()) != pad_value:
                continue

            for source_column in sources:
                value = int(nxt[row_index, source_column].item())
                if value != pad_value:
                    now[row_index, group_index] = value
                    nxt[row_index, source_column] = pad_value
                    moved += 1
                    break

    after = int((now != pad_value).sum().item()) + int(
        (nxt != pad_value).sum().item()
    )
    if before != after:
        raise AssertionError(
            f"CTF is not lossless: before={before}, after={after}"
        )

    return now, nxt, moved
