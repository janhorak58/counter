from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union
import math


def mae_rmse(pred, gt) -> Tuple[float, float]:
    """
    Accepts:
      - dict-like counts: {0: 10, 1: 2, ...} or {"0":10,...}
      - list/tuple vectors aligned by index: [tourist, skier, cyclist, dog]
    Returns: (MAE, RMSE)
    """
    # dict case
    if isinstance(pred, dict) and isinstance(gt, dict):
        # normalize keys to ints
        def norm(d: Dict) -> Dict[int, int]:
            out = {}
            for k, v in d.items():
                out[int(k)] = int(v)
            return out

        p = norm(pred)
        g = norm(gt)

        keys = sorted(set(g.keys()) | set(p.keys()))
        diffs = [(p.get(k, 0) - g.get(k, 0)) for k in keys]

    else:
        # vector/list case
        p_list = list(pred)  # type: ignore[arg-type]
        g_list = list(gt)    # type: ignore[arg-type]
        n = max(len(p_list), len(g_list))
        p_list = p_list + [0] * (n - len(p_list))
        g_list = g_list + [0] * (n - len(g_list))
        diffs = [p_list[i] - g_list[i] for i in range(n)]

    if not diffs:
        return 0.0, 0.0

    mae = sum(abs(d) for d in diffs) / len(diffs)
    rmse = math.sqrt(sum((d * d) for d in diffs) / len(diffs))
    return float(mae), float(rmse)
