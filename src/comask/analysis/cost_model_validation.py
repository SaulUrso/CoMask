from __future__ import annotations

import glob
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

import wandb


@dataclass
class RunValidationResult:
    run_id: str
    run_name: str
    run_path: str
    status: str
    skip_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    total_clients: Optional[int] = None
    clients_per_round: Optional[int] = None
    initial_model_size_bits: Optional[float] = None
    final_model_size_bits: Optional[float] = None
    prune_ratio: Optional[float] = None

    drop_rounds: List[int] = field(default_factory=list)
    step_probabilities: List[float] = field(default_factory=list)
    global_probability: Optional[float] = None

    rounds: List[int] = field(default_factory=list)
    actual_cumulative_cost: List[float] = field(default_factory=list)
    predicted_cumulative_global: List[float] = field(default_factory=list)
    predicted_cumulative_step: List[float] = field(default_factory=list)

    predicted_drop_rounds_global: List[float] = field(default_factory=list)
    predicted_drop_rounds_step: List[float] = field(default_factory=list)
    timing_metrics_global: Dict[str, Any] = field(default_factory=dict)
    timing_metrics_step: Dict[str, Any] = field(default_factory=dict)

    metrics_global: Dict[str, Any] = field(default_factory=dict)
    metrics_step: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepValidationResult:
    sweep_path: str
    gamma: int
    rho: float
    group_id: int
    proposal_cluster_id: Optional[int]
    results: List[RunValidationResult]


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_int(value: Any) -> Optional[int]:
    if not _is_finite_number(value):
        return None
    return int(float(value))


def _resolve_train_args(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict) and isinstance(config.get("train_args"), dict):
        return config["train_args"]
    return {}


def _find_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        if key in config and config[key] is not None:
            return config[key]

        for value in config.values():
            found = _find_config_value(value, key)
            if found is not None:
                return found

    return None


def _resolve_clients_per_round(total_clients: int, raw_value: Any) -> Optional[int]:
    if not _is_finite_number(raw_value):
        return None

    value = float(raw_value)
    if value <= 0:
        return None

    if value < 1:
        return max(1, int(round(total_clients * value)))

    return max(1, int(round(value)))


def _infer_clients_per_round_from_series(
    per_round_cost: Sequence[float],
    model_sizes: Sequence[float],
) -> Optional[int]:
    inferred_values: List[float] = []
    for cost_value, size_value in zip(per_round_cost, model_sizes):
        if not (_is_finite_number(cost_value) and _is_finite_number(size_value)):
            continue
        if float(size_value) <= 0:
            continue

        inferred = float(cost_value) / (2.0 * float(size_value))
        if inferred > 0:
            inferred_values.append(inferred)

    if not inferred_values:
        return None

    return max(1, int(round(float(np.median(np.array(inferred_values, dtype=float))))))


def _scan_round_series(
    run: Any,
    cost_key: str,
    model_size_key: str,
) -> Tuple[List[int], List[float], List[float], Optional[str]]:
    cost_by_round: Dict[int, float] = {}
    model_size_by_round: Dict[int, float] = {}

    for row in run.scan_history(keys=["round", cost_key, model_size_key]):
        round_value = _as_int(row.get("round"))
        if round_value is None:
            continue

        cost_value = row.get(cost_key)
        if _is_finite_number(cost_value):
            cost_by_round[round_value] = float(cost_value)

        model_size_value = row.get(model_size_key)
        if _is_finite_number(model_size_value):
            model_size_by_round[round_value] = float(model_size_value)

    if not cost_by_round:
        return [], [], [], "missing round-based communication cost values"

    rounds = sorted(cost_by_round)

    filled_model_sizes: List[float] = []
    last_size: Optional[float] = None
    for round_value in rounds:
        if round_value in model_size_by_round:
            last_size = model_size_by_round[round_value]
        filled_model_sizes.append(float("nan") if last_size is None else last_size)

    first_valid_idx = None
    for idx, value in enumerate(filled_model_sizes):
        if _is_finite_number(value):
            first_valid_idx = idx
            break

    if first_valid_idx is None:
        return [], [], [], "missing model size values"

    rounds = rounds[first_valid_idx:]
    costs = [cost_by_round[r] for r in rounds]
    sizes = filled_model_sizes[first_valid_idx:]
    return rounds, costs, sizes, None


def _detect_drop_rounds(
    rounds: Sequence[int],
    model_sizes: Sequence[float],
    tolerance: float = 1e-9,
) -> Tuple[List[int], Optional[str]]:
    drop_rounds: List[int] = []
    previous_size = float(model_sizes[0])

    for idx in range(1, len(model_sizes)):
        current_size = float(model_sizes[idx])
        if current_size > previous_size + tolerance:
            return (
                [],
                (f"model size increased at round {rounds[idx]} ({current_size} > {previous_size})"),
            )
        if current_size < previous_size - tolerance:
            drop_rounds.append(int(rounds[idx]))
        previous_size = current_size

    return drop_rounds, None


def _extract_table_rows(table_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = table_payload.get("columns", [])
    data = table_payload.get("data", [])

    rows: List[Dict[str, Any]] = []
    for row_data in data:
        if isinstance(row_data, list):
            row: Dict[str, Any] = {
                col: row_data[idx] if idx < len(row_data) else None for idx, col in enumerate(columns)
            }
        elif isinstance(row_data, dict):
            row = row_data
        else:
            continue

        round_value = _as_int(row.get("round"))
        if round_value is None:
            continue

        rows.append(
            {
                "round": round_value,
                "client_id": _as_int(row.get("client_id")),
                "cluster_id": _as_int(row.get("cluster_id")),
            }
        )

    return rows


def _load_mask_proposal_rows(
    run: Any,
    table_key: str = "mask_proposals",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    candidate_table_paths: List[str] = []

    summary_value = run.summary.get(table_key)
    if isinstance(summary_value, dict):
        table_path = summary_value.get("path")
        if isinstance(table_path, str):
            candidate_table_paths.append(table_path)

    if hasattr(run, "summary") and isinstance(run.summary, dict):
        for summary_key, summary_item in run.summary.items():
            if not isinstance(summary_item, dict):
                continue
            table_path = summary_item.get("path")
            if not isinstance(table_path, str):
                continue

            key_match = "mask" in str(summary_key).lower() or "proposal" in str(summary_key).lower()
            path_match = "mask" in table_path.lower() or "proposal" in table_path.lower()
            if key_match or path_match:
                candidate_table_paths.append(table_path)

    try:
        for file_ref in run.files():
            file_name = str(getattr(file_ref, "name", ""))
            lower_name = file_name.lower()
            if lower_name.endswith(".table.json") and ("mask" in lower_name and "proposal" in lower_name):
                candidate_table_paths.append(file_name)
    except Exception:
        pass

    deduped_table_paths: List[str] = []
    seen = set()
    for path in candidate_table_paths:
        if path not in seen:
            deduped_table_paths.append(path)
            seen.add(path)

    if not deduped_table_paths:
        return [], "mask_proposals table not found in summary or run files"

    table_errors: List[str] = []
    for table_path in deduped_table_paths:
        temp_dir = tempfile.mkdtemp(prefix="wandb-table-")
        try:
            file_ref = run.file(table_path)
            if file_ref is None:
                table_errors.append(f"unable to access {table_path}")
                continue

            downloaded = file_ref.download(root=temp_dir, replace=True)

            candidate_paths: List[str] = []
            if isinstance(downloaded, str):
                candidate_paths.append(downloaded)

            downloaded_name = getattr(downloaded, "name", None)
            if isinstance(downloaded_name, str):
                candidate_paths.append(downloaded_name)

            candidate_paths.append(os.path.join(temp_dir, table_path))
            candidate_paths.append(os.path.join(temp_dir, os.path.basename(table_path)))
            candidate_paths.extend(
                glob.glob(
                    os.path.join(temp_dir, "**", os.path.basename(table_path)),
                    recursive=True,
                )
            )

            for candidate_path in candidate_paths:
                if candidate_path and os.path.exists(candidate_path):
                    with open(candidate_path, "r", encoding="utf-8") as file_handle:
                        payload = json.load(file_handle)
                    rows = _extract_table_rows(payload)
                    if rows:
                        return rows, None
                    table_errors.append(f"table {table_path} has no valid rows")
                    break
            else:
                table_errors.append(f"downloaded {table_path} but could not locate local file")
        except Exception as exc:  # pragma: no cover - runtime/network specific
            table_errors.append(f"failed parsing {table_path}: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return [], "; ".join(table_errors) if table_errors else "failed to load mask proposal table"


def _build_intervals(rounds: Sequence[int], drop_rounds: Sequence[int], gamma: int) -> List[Tuple[int, int, List[int]]]:
    intervals: List[Tuple[int, int, List[int]]] = []
    interval_start = int(rounds[0])

    for idx in range(gamma):
        interval_end = int(drop_rounds[idx])
        interval_rounds = [r for r in rounds if interval_start <= r < interval_end]
        intervals.append((interval_start, interval_end, interval_rounds))
        interval_start = interval_end

    return intervals


def _estimate_step_probabilities(
    mask_rows: Sequence[Dict[str, Any]],
    intervals: Sequence[Tuple[int, int, List[int]]],
    clients_per_round: int,
    proposal_cluster_id: Optional[int],
) -> Tuple[List[float], List[Dict[str, Any]], Optional[str]]:
    filtered_rows = list(mask_rows)

    if proposal_cluster_id is not None:
        has_cluster_col = any(row.get("cluster_id") is not None for row in mask_rows)
        if has_cluster_col:
            filtered_rows = [row for row in mask_rows if row.get("cluster_id") == proposal_cluster_id]

    probabilities: List[float] = []
    step_stats: List[Dict[str, Any]] = []

    for step_idx, (start_round, end_round, interval_rounds) in enumerate(intervals):
        proposal_count = sum(1 for row in filtered_rows if start_round <= int(row["round"]) < end_round)
        rounds_in_step = len(interval_rounds)
        denominator = clients_per_round * rounds_in_step

        if denominator <= 0:
            return [], [], f"step {step_idx} has zero denominator for probability estimation"

        p_value = float(proposal_count) / float(denominator)
        probabilities.append(p_value)
        step_stats.append(
            {
                "step": step_idx,
                "start_round": start_round,
                "end_round": end_round,
                "num_rounds": rounds_in_step,
                "proposal_events": proposal_count,
                "denominator": denominator,
                "p_j": p_value,
            }
        )

    return probabilities, step_stats, None


def _predict_cumulative_cost(
    rounds: Sequence[int],
    total_clients: int,
    clients_per_round: int,
    initial_model_size_bits: float,
    prune_ratio: float,
    gamma: int,
    rho: float,
    global_p: Optional[float] = None,
    p_steps: Optional[Sequence[float]] = None,
    min_probability: float = 1e-12,
) -> np.ndarray:
    if p_steps is None and global_p is None:
        raise ValueError("either global_p or p_steps must be provided")

    if p_steps is not None and len(p_steps) != gamma:
        raise ValueError(f"expected {gamma} step probabilities, got {len(p_steps)}")

    v_req = max(1.0, float(rho) * float(total_clients))

    if p_steps is None:
        assert global_p is not None
        p_value = max(float(global_p), min_probability)
        delta_t_values = [v_req / (float(clients_per_round) * p_value)] * gamma
    else:
        delta_t_values = [v_req / (float(clients_per_round) * max(float(p), min_probability)) for p in p_steps]

    tau = [0.0]
    for delta_t in delta_t_values:
        tau.append(tau[-1] + float(delta_t))

    per_step_prune = float(prune_ratio) / float(gamma)
    q_values = [max(0.0, 1.0 - (j * per_step_prune)) for j in range(gamma + 1)]

    first_round = int(rounds[0])
    predicted: List[float] = []

    for round_value in rounds:
        t_value = float(int(round_value) - first_round + 1)

        weighted_duration = 0.0
        for step_idx in range(gamma + 1):
            if step_idx < gamma:
                duration = max(0.0, min(t_value, tau[step_idx + 1]) - tau[step_idx])
            else:
                duration = max(0.0, t_value - tau[gamma])
            weighted_duration += q_values[step_idx] * duration

        cumulative_cost = 2.0 * float(clients_per_round) * float(initial_model_size_bits) * weighted_duration
        predicted.append(cumulative_cost)

    return np.array(predicted, dtype=float)


def _predict_consolidation_rounds(
    rounds: Sequence[int],
    gamma: int,
    total_clients: int,
    clients_per_round: int,
    rho: float,
    global_p: Optional[float] = None,
    p_steps: Optional[Sequence[float]] = None,
    min_probability: float = 1e-12,
) -> List[float]:
    if len(rounds) == 0:
        return []

    if p_steps is None and global_p is None:
        raise ValueError("either global_p or p_steps must be provided")

    if p_steps is not None and len(p_steps) != gamma:
        raise ValueError(f"expected {gamma} step probabilities, got {len(p_steps)}")

    v_req = max(1.0, float(rho) * float(total_clients))

    if p_steps is None:
        assert global_p is not None
        p_value = max(float(global_p), min_probability)
        delta_t_values = [v_req / (float(clients_per_round) * p_value)] * gamma
    else:
        delta_t_values = [v_req / (float(clients_per_round) * max(float(p), min_probability)) for p in p_steps]

    tau = [0.0]
    for delta_t in delta_t_values:
        tau.append(tau[-1] + float(delta_t))

    first_round = int(rounds[0])
    return [float(first_round + tau[idx] - 1.0) for idx in range(1, gamma + 1)]


def _compute_timing_metrics(
    actual_drop_rounds: Sequence[int],
    predicted_drop_rounds: Sequence[float],
) -> Dict[str, Any]:
    observed_events = len(actual_drop_rounds)
    if observed_events == 0:
        return {
            "mae_rounds": float("nan"),
            "rmse_rounds": float("nan"),
            "mean_bias_rounds": float("nan"),
            "events_count": 0,
            "errors": [],
        }

    aligned_actual = np.array(actual_drop_rounds[:observed_events], dtype=float)
    aligned_pred = np.array(predicted_drop_rounds[:observed_events], dtype=float)
    errors = aligned_pred - aligned_actual

    mae_rounds = float(np.mean(np.abs(errors)))
    rmse_rounds = float(np.sqrt(np.mean(np.square(errors))))
    mean_bias_rounds = float(np.mean(errors))

    return {
        "mae_rounds": mae_rounds,
        "rmse_rounds": rmse_rounds,
        "mean_bias_rounds": mean_bias_rounds,
        "events_count": int(observed_events),
        "errors": errors.tolist(),
    }


def _build_segment_indices(rounds: Sequence[int], drop_rounds: Sequence[int]) -> np.ndarray:
    segment_indices = np.zeros(len(rounds), dtype=int)
    for idx, round_value in enumerate(rounds):
        segment_id = 0
        while segment_id < len(drop_rounds) and int(round_value) >= int(drop_rounds[segment_id]):
            segment_id += 1
        segment_indices[idx] = segment_id
    return segment_indices


def _compute_core_metrics(actual_cumulative: np.ndarray, predicted_cumulative: np.ndarray) -> Dict[str, float]:
    if actual_cumulative.size == 0 or predicted_cumulative.size == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "relative_rmse": float("nan"),
            "mape_percent": float("nan"),
            "smape_percent": float("nan"),
            "r2": float("nan"),
            "mean_bias": float("nan"),
            "mean_bias_percent": float("nan"),
            "relative_final_error": float("nan"),
            "final_bias_percent": float("nan"),
        }

    diff = predicted_cumulative - actual_cumulative
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))

    mean_abs_actual = float(np.mean(np.abs(actual_cumulative)))
    if mean_abs_actual < 1e-12:
        relative_rmse = float("nan")
    else:
        relative_rmse = float(rmse / mean_abs_actual)

    nonzero_mask = np.abs(actual_cumulative) > 1e-12
    if np.any(nonzero_mask):
        mape_percent = float(np.mean(np.abs(diff[nonzero_mask] / actual_cumulative[nonzero_mask])) * 100.0)
        mean_bias_percent = float(np.mean(diff[nonzero_mask] / actual_cumulative[nonzero_mask]) * 100.0)
    else:
        mape_percent = float("nan")
        mean_bias_percent = float("nan")

    smape_denom = np.abs(actual_cumulative) + np.abs(predicted_cumulative)
    smape_mask = smape_denom > 1e-12
    if np.any(smape_mask):
        smape_percent = float(np.mean(2.0 * np.abs(diff[smape_mask]) / smape_denom[smape_mask]) * 100.0)
    else:
        smape_percent = float("nan")

    ss_res = float(np.sum(np.square(diff)))
    actual_mean = float(np.mean(actual_cumulative))
    ss_tot = float(np.sum(np.square(actual_cumulative - actual_mean)))
    if ss_tot < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    mean_bias = float(np.mean(diff))

    final_actual = float(actual_cumulative[-1])
    if abs(final_actual) < 1e-12:
        relative_final_error = float("nan")
        final_bias_percent = float("nan")
    else:
        relative_final_error = float(diff[-1] / final_actual)
        final_bias_percent = float(relative_final_error * 100.0)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "mape_percent": mape_percent,
        "smape_percent": smape_percent,
        "r2": r2,
        "mean_bias": mean_bias,
        "mean_bias_percent": mean_bias_percent,
        "relative_final_error": relative_final_error,
        "final_bias_percent": final_bias_percent,
    }


def _compute_metrics(
    actual_cumulative: np.ndarray,
    predicted_cumulative: np.ndarray,
    segment_indices: Optional[np.ndarray] = None,
    segment_count: Optional[int] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = _compute_core_metrics(actual_cumulative, predicted_cumulative)

    if segment_indices is None:
        metrics["segments"] = {}
        return metrics

    if segment_count is None:
        segment_count = int(np.max(segment_indices)) + 1 if segment_indices.size > 0 else 0

    segment_metrics: Dict[str, Dict[str, float]] = {}
    for segment_id in range(segment_count):
        segment_mask = segment_indices == segment_id
        segment_metrics[f"segment_{segment_id}"] = _compute_core_metrics(
            actual_cumulative[segment_mask],
            predicted_cumulative[segment_mask],
        )

    metrics["segments"] = segment_metrics
    return metrics


def validate_run_cost_model(
    run: Any,
    gamma: int = 3,
    rho: float = 0.5,
    group_id: int = 0,
    proposal_cluster_id: Optional[int] = 0,
    max_rounds: Optional[int] = None,
) -> RunValidationResult:
    run_path = "/".join(run.path) if hasattr(run, "path") else run.id

    result = RunValidationResult(
        run_id=str(getattr(run, "id", "unknown")),
        run_name=str(getattr(run, "name", "unknown")),
        run_path=run_path,
        status="skipped",
    )

    run_config = dict(getattr(run, "config", {}))

    total_clients_raw = _find_config_value(run_config, "client_num_in_total")
    if not _is_finite_number(total_clients_raw):
        result.skip_reason = "missing train_args.client_num_in_total"
        return result

    total_clients = int(round(float(cast(float, total_clients_raw))))

    cost_key = f"CommCost/Group_{group_id}/Total/Combined"
    size_key = f"CommCost/Group_{group_id}/ModelSize/Bits"

    rounds, per_round_cost, model_sizes, series_error = _scan_round_series(run, cost_key, size_key)
    if series_error is not None:
        result.skip_reason = series_error
        return result

    clients_per_round = _resolve_clients_per_round(
        total_clients,
        _find_config_value(run_config, "client_num_per_round"),
    )
    if clients_per_round is None:
        clients_per_round = _infer_clients_per_round_from_series(per_round_cost, model_sizes)
        if clients_per_round is None:
            result.skip_reason = "missing client_num_per_round and unable to infer from round costs"
            return result
        result.warnings.append("client_num_per_round inferred from CommCost/Group_0/Total/Combined and model size")

    result.total_clients = total_clients
    result.clients_per_round = clients_per_round

    drop_rounds, monotonic_error = _detect_drop_rounds(rounds, model_sizes)
    if monotonic_error is not None:
        result.skip_reason = monotonic_error
        return result

    if len(drop_rounds) != gamma:
        result.skip_reason = f"expected exactly {gamma} model-size decreases, observed {len(drop_rounds)}"
        result.drop_rounds = drop_rounds
        return result

    initial_size = float(model_sizes[0])
    final_size = float(model_sizes[-1])
    if initial_size <= 0:
        result.skip_reason = "initial model size is not positive"
        return result

    prune_ratio = 1.0 - (final_size / initial_size)

    result.initial_model_size_bits = initial_size
    result.final_model_size_bits = final_size
    result.prune_ratio = prune_ratio
    result.drop_rounds = drop_rounds

    mask_rows, mask_error = _load_mask_proposal_rows(run)
    if mask_error is not None:
        result.skip_reason = mask_error
        return result

    intervals = _build_intervals(rounds, drop_rounds, gamma)
    p_steps, step_stats, probability_error = _estimate_step_probabilities(
        mask_rows=mask_rows,
        intervals=intervals,
        clients_per_round=clients_per_round,
        proposal_cluster_id=proposal_cluster_id,
    )
    if probability_error is not None:
        result.skip_reason = probability_error
        return result

    if any((not _is_finite_number(p)) for p in p_steps):
        result.skip_reason = "step probability estimation returned non-finite values"
        return result

    for stat in step_stats:
        if stat["p_j"] > 1.0:
            result.warnings.append(
                (
                    f"step {stat['step']} has p_j={stat['p_j']:.4f} > 1.0; "
                    "counting all proposal events may exceed one event per client-round"
                )
            )

    result.step_probabilities = p_steps
    result.global_probability = float(np.mean(p_steps))

    analysis_rounds = list(rounds)
    analysis_per_round_cost = list(per_round_cost)
    analysis_drop_rounds = list(drop_rounds)

    if max_rounds is not None:
        if max_rounds <= 0:
            result.skip_reason = "max_rounds must be a positive integer"
            return result

        analysis_rounds = analysis_rounds[: int(max_rounds)]
        analysis_per_round_cost = analysis_per_round_cost[: int(max_rounds)]
        if len(analysis_rounds) == 0:
            result.skip_reason = "no rounds available after applying max_rounds"
            return result

        max_analyzed_round = int(analysis_rounds[-1])
        analysis_drop_rounds = [int(r) for r in drop_rounds if int(r) <= max_analyzed_round]

    actual_cumulative = np.cumsum(np.array(analysis_per_round_cost, dtype=float))
    predicted_global = _predict_cumulative_cost(
        rounds=analysis_rounds,
        total_clients=total_clients,
        clients_per_round=clients_per_round,
        initial_model_size_bits=initial_size,
        prune_ratio=prune_ratio,
        gamma=gamma,
        rho=rho,
        global_p=result.global_probability,
    )
    predicted_step = _predict_cumulative_cost(
        rounds=analysis_rounds,
        total_clients=total_clients,
        clients_per_round=clients_per_round,
        initial_model_size_bits=initial_size,
        prune_ratio=prune_ratio,
        gamma=gamma,
        rho=rho,
        p_steps=p_steps,
    )

    predicted_drop_rounds_global = _predict_consolidation_rounds(
        rounds=analysis_rounds,
        gamma=gamma,
        total_clients=total_clients,
        clients_per_round=clients_per_round,
        rho=rho,
        global_p=result.global_probability,
    )
    predicted_drop_rounds_step = _predict_consolidation_rounds(
        rounds=analysis_rounds,
        gamma=gamma,
        total_clients=total_clients,
        clients_per_round=clients_per_round,
        rho=rho,
        p_steps=p_steps,
    )

    result.drop_rounds = analysis_drop_rounds
    result.rounds = [int(r) for r in analysis_rounds]
    result.actual_cumulative_cost = actual_cumulative.tolist()
    result.predicted_cumulative_global = predicted_global.tolist()
    result.predicted_cumulative_step = predicted_step.tolist()
    result.predicted_drop_rounds_global = predicted_drop_rounds_global
    result.predicted_drop_rounds_step = predicted_drop_rounds_step
    result.timing_metrics_global = _compute_timing_metrics(
        actual_drop_rounds=analysis_drop_rounds,
        predicted_drop_rounds=predicted_drop_rounds_global,
    )
    result.timing_metrics_step = _compute_timing_metrics(
        actual_drop_rounds=analysis_drop_rounds,
        predicted_drop_rounds=predicted_drop_rounds_step,
    )

    segment_indices = _build_segment_indices(analysis_rounds, analysis_drop_rounds)
    result.metrics_global = _compute_metrics(
        actual_cumulative,
        predicted_global,
        segment_indices=segment_indices,
        segment_count=gamma + 1,
    )
    result.metrics_step = _compute_metrics(
        actual_cumulative,
        predicted_step,
        segment_indices=segment_indices,
        segment_count=gamma + 1,
    )

    result.status = "evaluated"
    return result


def validate_sweep_cost_model(
    sweep_path: str,
    gamma: int = 3,
    rho: float = 0.5,
    group_id: int = 0,
    proposal_cluster_id: Optional[int] = 0,
    max_rounds: Optional[int] = None,
    api_timeout: int = 120,
) -> SweepValidationResult:
    api = wandb.Api(timeout=api_timeout)
    sweep = api.sweep(sweep_path)

    results: List[RunValidationResult] = []
    for run in sweep.runs:
        try:
            result = validate_run_cost_model(
                run=run,
                gamma=gamma,
                rho=rho,
                group_id=group_id,
                proposal_cluster_id=proposal_cluster_id,
                max_rounds=max_rounds,
            )
        except Exception as exc:  # pragma: no cover - runtime/network specific
            run_path = "/".join(run.path) if hasattr(run, "path") else run.id
            result = RunValidationResult(
                run_id=str(getattr(run, "id", "unknown")),
                run_name=str(getattr(run, "name", "unknown")),
                run_path=run_path,
                status="skipped",
                skip_reason=f"runtime failure: {exc}",
            )
        results.append(result)

    return SweepValidationResult(
        sweep_path=sweep_path,
        gamma=gamma,
        rho=rho,
        group_id=group_id,
        proposal_cluster_id=proposal_cluster_id,
        results=results,
    )


def build_summary_rows(validation_result: SweepValidationResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in validation_result.results:
        row = {
            "run_path": result.run_path,
            "status": result.status,
            "skip_reason": result.skip_reason,
            "total_clients": result.total_clients,
            "clients_per_round": result.clients_per_round,
            "initial_model_size_bits": result.initial_model_size_bits,
            "final_model_size_bits": result.final_model_size_bits,
            "prune_ratio": result.prune_ratio,
            "drop_rounds": result.drop_rounds,
            "p_global": result.global_probability,
            "p_step_0": result.step_probabilities[0] if len(result.step_probabilities) > 0 else None,
            "p_step_1": result.step_probabilities[1] if len(result.step_probabilities) > 1 else None,
            "p_step_2": result.step_probabilities[2] if len(result.step_probabilities) > 2 else None,
            "predicted_drop_rounds_global": result.predicted_drop_rounds_global,
            "predicted_drop_rounds_step": result.predicted_drop_rounds_step,
            "timing_mae_rounds_global": result.timing_metrics_global.get("mae_rounds"),
            "timing_rmse_rounds_global": result.timing_metrics_global.get("rmse_rounds"),
            "timing_mean_bias_rounds_global": result.timing_metrics_global.get("mean_bias_rounds"),
            "timing_events_count_global": result.timing_metrics_global.get("events_count"),
            "timing_mae_rounds_step": result.timing_metrics_step.get("mae_rounds"),
            "timing_rmse_rounds_step": result.timing_metrics_step.get("rmse_rounds"),
            "timing_mean_bias_rounds_step": result.timing_metrics_step.get("mean_bias_rounds"),
            "timing_events_count_step": result.timing_metrics_step.get("events_count"),
            "mae_global": result.metrics_global.get("mae"),
            "mse_global": result.metrics_global.get("mse"),
            "rmse_global": result.metrics_global.get("rmse"),
            "relative_rmse_global": result.metrics_global.get("relative_rmse"),
            "mape_percent_global": result.metrics_global.get("mape_percent"),
            "smape_percent_global": result.metrics_global.get("smape_percent"),
            "r2_global": result.metrics_global.get("r2"),
            "mean_bias_global": result.metrics_global.get("mean_bias"),
            "mean_bias_percent_global": result.metrics_global.get("mean_bias_percent"),
            "rel_final_err_global": result.metrics_global.get("relative_final_error"),
            "final_bias_percent_global": result.metrics_global.get("final_bias_percent"),
            "segments_global": result.metrics_global.get("segments"),
            "mae_step": result.metrics_step.get("mae"),
            "mse_step": result.metrics_step.get("mse"),
            "rmse_step": result.metrics_step.get("rmse"),
            "relative_rmse_step": result.metrics_step.get("relative_rmse"),
            "mape_percent_step": result.metrics_step.get("mape_percent"),
            "smape_percent_step": result.metrics_step.get("smape_percent"),
            "r2_step": result.metrics_step.get("r2"),
            "mean_bias_step": result.metrics_step.get("mean_bias"),
            "mean_bias_percent_step": result.metrics_step.get("mean_bias_percent"),
            "rel_final_err_step": result.metrics_step.get("relative_final_error"),
            "final_bias_percent_step": result.metrics_step.get("final_bias_percent"),
            "segments_step": result.metrics_step.get("segments"),
            "warnings": " | ".join(result.warnings),
        }
        rows.append(row)

    return rows


def build_consolidation_timing_rows(validation_result: SweepValidationResult) -> List[Dict[str, Any]]:
    evaluated_results = [result for result in validation_result.results if result.status == "evaluated"]

    def _rows_for_model(model_name: str, timing_attr: str) -> List[Dict[str, Any]]:
        collected_errors: List[float] = []
        run_mae_values: List[float] = []
        run_rmse_values: List[float] = []
        run_bias_values: List[float] = []
        runs_with_events = 0
        total_events = 0

        for result in evaluated_results:
            timing_metrics = getattr(result, timing_attr, {}) or {}
            events_count = int(timing_metrics.get("events_count", 0) or 0)
            if events_count <= 0:
                continue

            errors = timing_metrics.get("errors", []) or []
            errors_float = [float(error) for error in errors if _is_finite_number(error)]
            if len(errors_float) == 0:
                continue

            collected_errors.extend(errors_float)
            total_events += len(errors_float)
            runs_with_events += 1

            mae_rounds = timing_metrics.get("mae_rounds")
            rmse_rounds = timing_metrics.get("rmse_rounds")
            mean_bias_rounds = timing_metrics.get("mean_bias_rounds")

            if _is_finite_number(mae_rounds):
                run_mae_values.append(float(mae_rounds))
            if _is_finite_number(rmse_rounds):
                run_rmse_values.append(float(rmse_rounds))
            if _is_finite_number(mean_bias_rounds):
                run_bias_values.append(float(mean_bias_rounds))

        if len(collected_errors) > 0:
            errors_array = np.array(collected_errors, dtype=float)
            micro_mae = float(np.mean(np.abs(errors_array)))
            micro_rmse = float(np.sqrt(np.mean(np.square(errors_array))))
            micro_bias = float(np.mean(errors_array))
        else:
            micro_mae = float("nan")
            micro_rmse = float("nan")
            micro_bias = float("nan")

        macro_mae = float(np.mean(run_mae_values)) if len(run_mae_values) > 0 else float("nan")
        macro_rmse = float(np.mean(run_rmse_values)) if len(run_rmse_values) > 0 else float("nan")
        macro_bias = float(np.mean(run_bias_values)) if len(run_bias_values) > 0 else float("nan")

        return [
            {
                "model": model_name,
                "aggregation": "micro",
                "mae_rounds": micro_mae,
                "rmse_rounds": micro_rmse,
                "mean_bias_rounds": micro_bias,
                "runs_count": runs_with_events,
                "events_count": total_events,
            },
            {
                "model": model_name,
                "aggregation": "macro",
                "mae_rounds": macro_mae,
                "rmse_rounds": macro_rmse,
                "mean_bias_rounds": macro_bias,
                "runs_count": runs_with_events,
                "events_count": total_events,
            },
        ]

    rows: List[Dict[str, Any]] = []
    rows.extend(_rows_for_model("global_probability", "timing_metrics_global"))
    rows.extend(_rows_for_model("step_probability", "timing_metrics_step"))
    return rows
