"""Focused causal audit of Jeans Kalman parameter policies.

This file is executed from research/research.ipynb after the earlier Jeans
cells.  It deliberately receives the already loaded Round 1 price array from
the notebook and never reads prices from a simulated strategy.  Every state,
score, refit, and position is a prefix-only calculation.
"""

import math
from collections import Counter


AUDIT_PRICE = np.asarray(JEANS_PRICE, dtype=float)
AUDIT_N = len(AUDIT_PRICE)
pd.set_option('display.max_rows', 200)
AUDIT_LIMIT = int(JEANS_LIMIT)
AUDIT_ALWAYS_LONG = np.full(AUDIT_N, AUDIT_LIMIT, dtype=int)

# These are frozen before inspecting the new results.  They are the K1/K2/K3
# values specified in the audit brief, rather than the separate robustness
# surface used in the preceding follow-up.
AUDIT_KALMAN_SPECS = {
    'K1': (0.25, 0.02, 9.0),
    'K2': (1.00, 0.05, 9.0),
    'K3': (2.00, 0.10, 16.0),
}
AUDIT_GATES = (0.75, 1.0, 1.25, 1.5)
AUDIT_INITIAL_SLOPE_VARIANCES = (0.25, 1.0, 4.0)
AUDIT_MAPPING_SHORTS = {
    'full_short': -800,
    'conservative_short': -400,
    'long_flat': 0,
}
AUDIT_SELECTOR_WARMUPS = (60, 90)
AUDIT_SELECTOR_SCORING = ('expanding', 'trailing60', 'trailing120')
AUDIT_REFIT_INTERVAL = 30
AUDIT_MLE_BOUNDS = np.log(np.asarray([[0.10, 4.00], [0.005, 0.20], [4.0, 25.0]], dtype=float))
AUDIT_MLE_STARTS = {
    'K2': np.log(np.asarray(AUDIT_KALMAN_SPECS['K2'], dtype=float)),
    'K1': np.log(np.asarray(AUDIT_KALMAN_SPECS['K1'], dtype=float)),
    'K3': np.log(np.asarray(AUDIT_KALMAN_SPECS['K3'], dtype=float)),
    'geometric_mid': np.mean(AUDIT_MLE_BOUNDS, axis=1),
}


def audit_kalman_filter(price, parameters, initial_slope_variance=1.0, transform='raw'):
    """Return causal local-linear slope and uncertainty arrays."""
    price = np.asarray(price, dtype=float)
    if len(price) == 0:
        return np.empty(0), np.empty(0)
    q_level, q_slope, observation_var = [float(x) for x in parameters]
    observation = np.log(np.maximum(price, 1e-9)) if transform == 'log' else price
    transition = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([q_level, q_slope])
    H = np.asarray([1.0, 0.0])
    state = np.asarray([observation[0], 0.0], dtype=float)
    covariance = np.diag([observation_var, float(initial_slope_variance)])
    slopes = np.zeros(len(price), dtype=float)
    uncertainties = np.zeros(len(price), dtype=float)
    for day in range(len(price)):
        if day > 0:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        innovation = float(observation[day] - H @ state)
        innovation_var = float(H @ covariance @ H.T) + observation_var
        gain = covariance @ H / max(innovation_var, 1e-12)
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, H @ covariance)
        covariance = (covariance + covariance.T) / 2.0
        slopes[day] = state[1]
        uncertainties[day] = math.sqrt(max(float(covariance[1, 1]), 0.0))
    return slopes, uncertainties


def audit_kalman_state(price, parameters, gate=1.0, initial_slope_variance=1.0, transform='raw'):
    slope, uncertainty = audit_kalman_filter(price, parameters, initial_slope_variance, transform)
    state = np.where(slope < -float(gate) * uncertainty, -1, 1).astype(int)
    state[0] = 0
    return state, slope, uncertainty


def audit_position_from_state(state, short_position=-800, zero_first=True):
    state = np.asarray(state, dtype=int)
    position = np.where(state < 0, int(short_position), AUDIT_LIMIT).astype(int)
    if zero_first and len(position):
        position[0] = 0
    return position


def audit_loglik(price, parameters, start=0, end=None, initial_slope_variance=1.0, transform='raw'):
    """One-step Gaussian predictive log likelihood over a visible prefix."""
    price = np.asarray(price, dtype=float)
    end = len(price) if end is None else min(int(end), len(price))
    start = max(0, int(start))
    if end - start <= 1:
        return 0.0
    q_level, q_slope, observation_var = [float(x) for x in parameters]
    observation = np.log(np.maximum(price, 1e-9)) if transform == 'log' else price
    transition = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([q_level, q_slope])
    H = np.asarray([1.0, 0.0])
    state = np.asarray([observation[start], 0.0], dtype=float)
    covariance = np.diag([observation_var, float(initial_slope_variance)])
    log_likelihood = 0.0
    for day in range(start + 1, end):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        innovation = float(observation[day] - H @ state)
        innovation_var = max(float(H @ covariance @ H.T) + observation_var, 1e-12)
        log_likelihood += -0.5 * (math.log(2.0 * math.pi * innovation_var) + innovation * innovation / innovation_var)
        gain = covariance @ H / innovation_var
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, H @ covariance)
        covariance = (covariance + covariance.T) / 2.0
    return float(log_likelihood)


def audit_score_window(price, parameters, refit_day, scoring):
    if scoring == 'expanding':
        start = 0
    elif scoring == 'trailing60':
        start = max(0, int(refit_day) - 60)
    elif scoring == 'trailing120':
        start = max(0, int(refit_day) - 120)
    else:
        raise ValueError(scoring)
    # The endpoint is exclusive.  At refit day r, price[r] is not scored.
    return audit_loglik(price, parameters, start=start, end=refit_day)


def audit_selector_positions(price, warmup, scoring, every=AUDIT_REFIT_INTERVAL):
    """Causally select K1/K2/K3, then map the selected filter to +800/-800."""
    price = np.asarray(price, dtype=float)
    state_paths = {
        name: audit_kalman_state(price, parameters, gate=1.0)[0]
        for name, parameters in AUDIT_KALMAN_SPECS.items()
    }
    refit_days = list(range(int(warmup), len(price), int(every)))
    refit_lookup = set(refit_days)
    selected = 'K2'
    position = np.zeros(len(price), dtype=int)
    records = []
    for day in range(len(price)):
        if day in refit_lookup:
            scores = {name: audit_score_window(price, parameters, day, scoring) for name, parameters in AUDIT_KALMAN_SPECS.items()}
            # Stable tie break: K2 is preferred over K1, which is preferred over K3.
            selected = max(('K2', 'K1', 'K3'), key=lambda name: (scores[name], {'K2': 2, 'K1': 1, 'K3': 0}[name]))
            records.append({
                'Refit day': day,
                'Warm-up': int(warmup),
                'Scoring': scoring,
                'Selected': selected,
                'K1 predictive LL': scores['K1'],
                'K2 predictive LL': scores['K2'],
                'K3 predictive LL': scores['K3'],
            })
        position[day] = -AUDIT_LIMIT if state_paths[selected][day] < 0 else AUDIT_LIMIT
    position[0] = 0
    return position, pd.DataFrame(records)


def audit_pattern_search(objective, start, bounds, max_iter=4):
    """Small deterministic bounded optimizer in log-variance space.

    This is intentionally low capacity: the same four starts and coordinate
    pattern are used at every online refit.  It is an optimizer for the audit,
    not a fitted full-sample parameter set.
    """
    lower = np.asarray(bounds[:, 0], dtype=float)
    upper = np.asarray(bounds[:, 1], dtype=float)
    theta = np.clip(np.asarray(start, dtype=float), lower, upper)
    value = float(objective(theta))
    if not np.isfinite(value):
        return {'theta': theta, 'objective': np.inf, 'success': False, 'iterations': 0}
    step = (upper - lower) * 0.25
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        improved = False
        for coordinate in range(len(theta)):
            best_theta = theta.copy()
            best_value = value
            for direction in (-1.0, 1.0):
                candidate = theta.copy()
                candidate[coordinate] = np.clip(candidate[coordinate] + direction * step[coordinate], lower[coordinate], upper[coordinate])
                candidate_value = float(objective(candidate))
                if np.isfinite(candidate_value) and candidate_value < best_value - 1e-9:
                    best_theta, best_value = candidate, candidate_value
            if best_value < value - 1e-9:
                theta, value = best_theta, best_value
                improved = True
        if not improved:
            step *= 0.5
        if float(np.max(step)) < 1e-3:
            break
    return {
        'theta': theta,
        'objective': float(value),
        'success': bool(np.isfinite(value)),
        'iterations': int(iterations),
    }


def audit_fit_mle(price, refit_day, scoring):
    """Fit bounded log variances using only observations before refit_day."""
    price = np.asarray(price, dtype=float)
    if scoring == 'expanding':
        start = 0
    elif scoring == 'trailing120':
        start = max(0, int(refit_day) - 120)
    else:
        raise ValueError(scoring)
    start_records = []
    for start_name, start_theta in AUDIT_MLE_STARTS.items():
        def objective(theta):
            return -audit_loglik(price, np.exp(theta), start=start, end=refit_day)
        try:
            result = audit_pattern_search(objective, start_theta, AUDIT_MLE_BOUNDS)
            row = {
                'Refit day': int(refit_day),
                'Scoring': scoring,
                'Optimizer start': start_name,
                'Success': bool(result['success']),
                'Objective': float(result['objective']),
                'Iterations': int(result['iterations']),
                'q_level': float(np.exp(result['theta'][0])),
                'q_slope': float(np.exp(result['theta'][1])),
                'R': float(np.exp(result['theta'][2])),
            }
        except Exception as exc:
            row = {
                'Refit day': int(refit_day), 'Scoring': scoring,
                'Optimizer start': start_name, 'Success': False,
                'Objective': np.inf, 'Iterations': 0,
                'q_level': np.nan, 'q_slope': np.nan, 'R': np.nan,
                'Error': repr(exc),
            }
        start_records.append(row)
    successful = [row for row in start_records if row['Success'] and np.isfinite(row['Objective'])]
    if successful:
        best = min(successful, key=lambda row: row['Objective'])
        parameters = (best['q_level'], best['q_slope'], best['R'])
        boundary = any(
            abs(math.log(parameters[i]) - AUDIT_MLE_BOUNDS[i, 0]) < 1e-3
            or abs(math.log(parameters[i]) - AUDIT_MLE_BOUNDS[i, 1]) < 1e-3
            for i in range(3)
        )
        return parameters, best, start_records, int(not all(row['Success'] for row in start_records)), bool(boundary)
    fallback = AUDIT_KALMAN_SPECS['K2']
    return fallback, {
        'Refit day': int(refit_day), 'Scoring': scoring,
        'Optimizer start': 'fallback_K2', 'Success': False,
        'Objective': np.inf, 'Iterations': 0,
        'q_level': fallback[0], 'q_slope': fallback[1], 'R': fallback[2],
    }, start_records, 1, False


def audit_mle_positions(price, warmup, scoring, every=AUDIT_REFIT_INTERVAL):
    """Online MLE policy with causal refits and visible-history reconstruction."""
    price = np.asarray(price, dtype=float)
    refit_days = list(range(int(warmup), len(price), int(every)))
    records = []
    all_start_records = []
    fits = {}
    previous_parameters = None
    for refit_day in refit_days:
        parameters, best, starts, failures, boundary = audit_fit_mle(price, refit_day, scoring)
        slopes, uncertainties = audit_kalman_filter(price, parameters)
        slope_z = float(slopes[refit_day] / uncertainties[refit_day]) if uncertainties[refit_day] > 0 else np.nan
        parameter_jump = np.nan if previous_parameters is None else float(np.max(np.abs(np.log(np.asarray(parameters) / np.asarray(previous_parameters)))))
        record = {
            'Refit day': int(refit_day), 'Warm-up': int(warmup), 'Scoring': scoring,
            'Selected optimizer start': best.get('Optimizer start', ''),
            'Optimizer success': bool(best.get('Success', False)),
            'Optimizer failures': int(failures),
            'Boundary solution': bool(boundary),
            'q_level': float(parameters[0]), 'q_slope': float(parameters[1]), 'R': float(parameters[2]),
            'q_level/R': float(parameters[0] / parameters[2]),
            'q_slope/R': float(parameters[1] / parameters[2]),
            'Visible-history log likelihood': float(-best.get('Objective', np.inf)),
            'Slope z-score': slope_z,
            'Selected position': int(-AUDIT_LIMIT if slope_z < -1.0 else AUDIT_LIMIT),
            'Max log-parameter jump': parameter_jump,
        }
        records.append(record)
        all_start_records.extend(starts)
        fits[refit_day] = (parameters, slopes, uncertainties)
        previous_parameters = parameters
    position = np.zeros(len(price), dtype=int)
    # K2 is the declared startup filter until the first refit.
    startup_state = audit_kalman_state(price, AUDIT_KALMAN_SPECS['K2'])[0]
    position[:refit_days[0] if refit_days else len(price)] = audit_position_from_state(startup_state[:refit_days[0] if refit_days else len(price)], -AUDIT_LIMIT)
    for index, refit_day in enumerate(refit_days):
        end = refit_days[index + 1] if index + 1 < len(refit_days) else len(price)
        parameters, slopes, uncertainties = fits[refit_day]
        state = np.where(slopes < -uncertainties, -1, 1).astype(int)
        state[0] = 0
        position[refit_day:end] = audit_position_from_state(state[refit_day:end], -AUDIT_LIMIT, zero_first=False)
    position[0] = 0
    records_frame = pd.DataFrame(records)
    if not records_frame.empty:
        for index, row in records_frame.iterrows():
            block_end = int(records_frame.iloc[index + 1]['Refit day']) if index + 1 < len(records_frame) else len(price)
            records_frame.loc[index, 'Subsequent block P&L'] = float(np.sum(realized_pnl(price, position)[int(row['Refit day']):block_end]))
    return position, records_frame, pd.DataFrame(all_start_records)


def audit_normalized_filter(price, decay):
    """Causal EWMA scale filter preserving K2 q/R ratios."""
    price = np.asarray(price, dtype=float)
    if len(price) == 0:
        return np.empty(0), np.empty(0)
    q_level_ratio = AUDIT_KALMAN_SPECS['K2'][0] / AUDIT_KALMAN_SPECS['K2'][2]
    q_slope_ratio = AUDIT_KALMAN_SPECS['K2'][1] / AUDIT_KALMAN_SPECS['K2'][2]
    scale_variance = max((0.01 * abs(price[0])) ** 2, 1e-8)
    state = np.asarray([price[0], 0.0], dtype=float)
    covariance = np.diag([scale_variance, scale_variance])
    transition = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    H = np.asarray([1.0, 0.0])
    slopes = np.zeros(len(price), dtype=float)
    uncertainties = np.zeros(len(price), dtype=float)
    for day in range(len(price)):
        if day > 0:
            change = price[day] - price[day - 1]
            scale_variance = float(decay) * scale_variance + (1.0 - float(decay)) * change * change
            scale_variance = max(scale_variance, 1e-8)
            state = transition @ state
            process = np.diag([q_level_ratio * scale_variance, q_slope_ratio * scale_variance])
            covariance = transition @ covariance @ transition.T + process
        observation_var = scale_variance
        innovation = float(price[day] - H @ state)
        innovation_var = max(float(H @ covariance @ H.T) + observation_var, 1e-12)
        gain = covariance @ H / innovation_var
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, H @ covariance)
        covariance = (covariance + covariance.T) / 2.0
        slopes[day] = state[1]
        uncertainties[day] = math.sqrt(max(float(covariance[1, 1]), 0.0))
    return slopes, uncertainties


def audit_position_from_slope(slopes, uncertainties, short=-800):
    position = np.where(np.asarray(slopes) < -np.asarray(uncertainties), int(short), AUDIT_LIMIT).astype(int)
    position[0] = 0
    return position


def audit_add(candidates, records, name, family, position, parameters):
    position = np.asarray(position, dtype=int)
    assert len(position) == AUDIT_N
    assert np.issubdtype(position.dtype, np.integer)
    assert np.max(np.abs(position)) <= AUDIT_LIMIT
    candidates[name] = position
    records.append({'Candidate': name, 'Family': family, 'Parameters': parameters})


def build_audit_fixed_policies(price):
    """Build the predeclared fixed, consensus, ensemble, and surface grids."""
    price = np.asarray(price, dtype=float)
    candidates = {}
    records = []
    state_cache = {}
    for label, parameters in AUDIT_KALMAN_SPECS.items():
        for gate in AUDIT_GATES:
            state_cache[(label, gate, 1.0)] = audit_kalman_state(price, parameters, gate=gate, initial_slope_variance=1.0)[0]
            for mapping, short in AUDIT_MAPPING_SHORTS.items():
                position = audit_position_from_state(state_cache[(label, gate, 1.0)], short)
                audit_add(candidates, records, f'fixed_{label}_c{str(gate).replace(".", "p")}_{mapping}', 'Fixed Kalman', position, f'{label}; c={gate}; +800 else {short}')
    for label, parameters in AUDIT_KALMAN_SPECS.items():
        for initial_variance in AUDIT_INITIAL_SLOPE_VARIANCES:
            for gate in AUDIT_GATES:
                state = audit_kalman_state(price, parameters, gate=gate, initial_slope_variance=initial_variance)[0]
                position = audit_position_from_state(state, -800)
                name = f'kalman_surface_{label}_iv{str(initial_variance).replace(".", "p")}_c{str(gate).replace(".", "p")}'
                audit_add(candidates, records, name, 'Kalman robustness surface', position, f'{label}; initial slope variance={initial_variance}; c={gate}; full short')
    k1_state = audit_kalman_state(price, AUDIT_KALMAN_SPECS['K1'], gate=1.0)[0]
    k2_state = audit_kalman_state(price, AUDIT_KALMAN_SPECS['K2'], gate=1.0)[0]
    k3_state = audit_kalman_state(price, AUDIT_KALMAN_SPECS['K3'], gate=1.0)[0]
    consensus_down = (k1_state < 0) & (k2_state < 0)
    ensemble_down = ((k1_state < 0).astype(int) + (k2_state < 0).astype(int) + (k3_state < 0).astype(int)) >= 2
    for label, mask in (('K1_K2_consensus', consensus_down), ('K1_K2_K3_ensemble', ensemble_down)):
        for short_name, short in (('full_short', -800), ('conservative_short', -400)):
            position = np.where(mask, short, AUDIT_LIMIT).astype(int)
            position[0] = 0
            family = 'Fixed K1/K2 consensus' if label.startswith('K1_K2_consensus') else 'Fixed K1/K2/K3 ensemble'
            audit_add(candidates, records, f'{label}_{short_name}', family, position, f'c=1; +800 else {short}')
    return candidates, pd.DataFrame(records).set_index('Candidate')


def build_audit_adaptive_policies(price, include_mle=True, mle_warmups=None, mle_scoring=None):
    price = np.asarray(price, dtype=float)
    candidates = {}
    records = []
    detail = {}
    for warmup in AUDIT_SELECTOR_WARMUPS:
        for scoring in AUDIT_SELECTOR_SCORING:
            name = f'selector_w{warmup}_{scoring}'
            position, selection = audit_selector_positions(price, warmup, scoring)
            audit_add(candidates, records, name, 'Causal discrete selector', position, f'K1/K2/K3; warm-up={warmup}; refit=30; {scoring}; predictive LL')
            detail[name] = {'selection': selection}
    if include_mle:
        mle_warmups = AUDIT_SELECTOR_WARMUPS if mle_warmups is None else tuple(mle_warmups)
        mle_scoring = ('expanding', 'trailing120') if mle_scoring is None else tuple(mle_scoring)
        for warmup in mle_warmups:
            for scoring in mle_scoring:
                name = f'online_mle_w{warmup}_{scoring}'
                position, fits, starts = audit_mle_positions(price, warmup, scoring)
                audit_add(candidates, records, name, 'Constrained online MLE', position, f'log variances; warm-up={warmup}; refit=30; {scoring}; c=1')
                detail[name] = {'fits': fits, 'starts': starts}
    log_parameters = (0.00030, 0.00002, 0.00250)
    log_state = audit_kalman_state(price, log_parameters, gate=1.0, initial_slope_variance=0.0001, transform='log')[0]
    audit_add(candidates, records, 'scale_log_K2', 'Scale-robust filter', audit_position_from_state(log_state, -800), 'log price; K2-scale check; c=1')
    for decay in (0.90, 0.98):
        slopes, uncertainty = audit_normalized_filter(price, decay)
        audit_add(candidates, records, f'scale_ewma_{str(decay).replace(".", "p")}', 'Scale-robust filter', audit_position_from_slope(slopes, uncertainty, -800), f'causal EWMA change variance decay={decay}; K2 q/R ratios; c=1')
    return candidates, pd.DataFrame(records).set_index('Candidate'), detail


def audit_segment_metrics(price, position):
    pnl = realized_pnl(price, position)
    long_mask = np.zeros(len(price), dtype=bool)
    short_mask = np.zeros(len(price), dtype=bool)
    long_mask[1:] = np.asarray(position[:-1]) > 0
    short_mask[1:] = np.asarray(position[:-1]) < 0
    row = strategy_metrics(price, position)
    row.update({
        'Long P&L': float(np.sum(pnl[long_mask])),
        'Short P&L': float(np.sum(pnl[short_mask])),
        'Long days': int(np.sum(position > 0)),
        'Short days': int(np.sum(position < 0)),
        'Flat days': int(np.sum(position == 0)),
        'Q1': float(np.sum(pnl[0:91])), 'Q2': float(np.sum(pnl[91:182])),
        'Q3': float(np.sum(pnl[182:273])), 'Q4': float(np.sum(pnl[273:364])),
        'H1': float(np.sum(pnl[0:182])), 'H2': float(np.sum(pnl[182:364])),
        'P&L excl final 30': float(np.sum(pnl[:-30])),
        'P&L excl final 60': float(np.sum(pnl[:-60])),
        'P&L excl final 91': float(np.sum(pnl[:-91])),
        'P&L after best 5': float(np.sum(pnl) - np.sum(np.sort(pnl)[-5:])),
        'Negative K2 regimes': int(len(state_run_lengths(REGIME_STATES['K2 confidence'], -1))),
    })
    return row


def audit_block_pnl(price, position, boundaries):
    pnl = realized_pnl(price, position)
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows.append({'Start': int(start), 'End': int(end), 'P&L': float(np.sum(pnl[start:end]))})
    return pd.DataFrame(rows)


AUDIT_FIXED_CANDIDATES, AUDIT_FIXED_CATALOG = build_audit_fixed_policies(AUDIT_PRICE)
AUDIT_ADAPTIVE_CANDIDATES, AUDIT_ADAPTIVE_CATALOG, AUDIT_ADAPTIVE_DETAIL = build_audit_adaptive_policies(AUDIT_PRICE, include_mle=True)
AUDIT_CANDIDATES = {}
AUDIT_CANDIDATES.update(AUDIT_FIXED_CANDIDATES)
AUDIT_CANDIDATES.update(AUDIT_ADAPTIVE_CANDIDATES)
AUDIT_CATALOG = pd.concat([AUDIT_FIXED_CATALOG, AUDIT_ADAPTIVE_CATALOG], axis=0)

AUDIT_ALWAYS_PNL = float(np.sum(realized_pnl(AUDIT_PRICE, AUDIT_ALWAYS_LONG)))
AUDIT_METRIC_ROWS = []
for name, position in [('always_long', AUDIT_ALWAYS_LONG), *list(AUDIT_CANDIDATES.items())]:
    row = audit_segment_metrics(AUDIT_PRICE, position)
    row.update({'Candidate': name, 'Family': 'Baseline' if name == 'always_long' else AUDIT_CATALOG.loc[name, 'Family'], 'Parameters': 'structural +800' if name == 'always_long' else AUDIT_CATALOG.loc[name, 'Parameters']})
    row['Incremental vs always long'] = row['P&L'] - AUDIT_ALWAYS_PNL
    AUDIT_METRIC_ROWS.append(row)
AUDIT_TABLE = pd.DataFrame(AUDIT_METRIC_ROWS).set_index('Candidate')

AUDIT_SERIOUS_NAMES = [
    'always_long',
    'fixed_K1_c1p0_full_short', 'fixed_K2_c1p0_full_short', 'fixed_K3_c1p0_full_short',
    'fixed_K1_c1p0_conservative_short', 'fixed_K2_c1p0_conservative_short', 'fixed_K3_c1p0_conservative_short',
    'fixed_K1_c1p0_long_flat', 'fixed_K2_c1p0_long_flat', 'fixed_K3_c1p0_long_flat',
    'K1_K2_consensus_full_short', 'K1_K2_consensus_conservative_short',
    'K1_K2_K3_ensemble_full_short', 'K1_K2_K3_ensemble_conservative_short',
    'selector_w60_expanding', 'selector_w90_expanding', 'selector_w60_trailing60', 'selector_w90_trailing120',
    'online_mle_w60_expanding', 'online_mle_w90_expanding', 'online_mle_w60_trailing120', 'online_mle_w90_trailing120',
    'scale_log_K2', 'scale_ewma_0p9', 'scale_ewma_0p98',
]
AUDIT_SERIOUS_NAMES = [name for name in AUDIT_SERIOUS_NAMES if name in AUDIT_TABLE.index]
AUDIT_COMPARISON_COLUMNS = [
    'Family', 'P&L', 'Incremental vs always long', 'Daily mean / sd', 'Sharpe ann.',
    'Hit rate active', 'Active days', 'Turnover units', 'Max drawdown',
    'Max Jeans exposure AUD', 'Longest losing streak', 'Long P&L', 'Short P&L',
    'Q1', 'Q2', 'Q3', 'Q4', 'H1', 'H2', 'P&L excl final 91', 'P&L after best 5',
]
print('Focused policy search counts (frozen before results):')
display(pd.DataFrame([
    {'Family': 'Fixed K1/K2/K3 x 4 gates x 3 mappings', 'Candidates': 36},
    {'Family': 'Fixed K1/K2/K3 robustness surface x 3 initial variances x 4 gates', 'Candidates': 36},
    {'Family': 'Fixed K1/K2 consensus', 'Candidates': 2},
    {'Family': 'Fixed K1/K2/K3 equal vote', 'Candidates': 2},
    {'Family': 'Causal discrete selectors', 'Candidates': 6},
    {'Family': 'Constrained online MLE', 'Candidates': 4},
    {'Family': 'Scale-robust alternatives', 'Candidates': 3},
    {'Family': 'Total new candidates', 'Candidates': len(AUDIT_CANDIDATES)},
    {'Family': 'Earlier combined catalogue', 'Candidates': len(FOLLOWUP_CANDIDATES)},
],).set_index('Family'))
print('Every audit position is integer and within the 800-unit Jeans limit.')
display(AUDIT_TABLE.loc[AUDIT_SERIOUS_NAMES, AUDIT_COMPARISON_COLUMNS].round(3))

# Parameter surfaces are shown as surfaces, not as a winner-take-all selection.
surface_rows = []
for label in AUDIT_KALMAN_SPECS:
    names = [name for name in AUDIT_CANDIDATES if name.startswith('kalman_surface_' + label + '_')]
    for name in names:
        row = AUDIT_TABLE.loc[name]
        surface_rows.append({
            'Spec': label,
            'Initial slope variance': AUDIT_CATALOG.loc[name, 'Parameters'].split('initial slope variance=')[1].split(';')[0],
            'Gate': AUDIT_CATALOG.loc[name, 'Parameters'].split('c=')[1].split(';')[0],
            'P&L': row['P&L'], 'Incremental': row['Incremental vs always long'],
            'Q1': row['Q1'], 'Q2': row['Q2'], 'Q3': row['Q3'], 'Q4': row['Q4'],
        })
AUDIT_KALMAN_SURFACE = pd.DataFrame(surface_rows)
print('Raw-price Kalman robustness surface: full-short mapping, all frozen gates and initial variances.')
display(AUDIT_KALMAN_SURFACE.round(2))

# Selection records and subsequent blocks are kept explicitly in notebook output.
AUDIT_SELECTOR_RECORDS = []
for name, detail in AUDIT_ADAPTIVE_DETAIL.items():
    if 'selection' in detail:
        selection = detail['selection'].copy()
        if selection.empty:
            continue
        position = AUDIT_CANDIDATES[name]
        pnl = realized_pnl(AUDIT_PRICE, position)
        selection['Subsequent block end'] = selection['Refit day'].shift(-1).fillna(AUDIT_N).astype(int)
        selection['Subsequent block P&L'] = [float(np.sum(pnl[int(start):int(end)])) for start, end in zip(selection['Refit day'], selection['Subsequent block end'])]
        selection['Policy'] = name
        selection['Selection changed from prior'] = selection['Selected'].ne(selection['Selected'].shift(1)).fillna(False)
        AUDIT_SELECTOR_RECORDS.append(selection)
AUDIT_SELECTOR_BLOCKS = pd.concat(AUDIT_SELECTOR_RECORDS, ignore_index=True) if AUDIT_SELECTOR_RECORDS else pd.DataFrame()
print('Causal discrete selection records: scores use only prices before each refit; P&L is subsequent to that selection.')
display(AUDIT_SELECTOR_BLOCKS.round(3))
if not AUDIT_SELECTOR_BLOCKS.empty:
    display(AUDIT_SELECTOR_BLOCKS.groupby('Policy').agg(
        Refits=('Refit day', 'count'), Changes=('Selection changed from prior', 'sum'),
        Mean_subsequent_block_PnL=('Subsequent block P&L', 'mean'), Total_subsequent_block_PnL=('Subsequent block P&L', 'sum')
    ).round(3))

AUDIT_MLE_PARAMETER_TABLE = pd.concat([
    detail['fits'].assign(Policy=name) for name, detail in AUDIT_ADAPTIVE_DETAIL.items() if 'fits' in detail and not detail['fits'].empty
], ignore_index=True)
AUDIT_MLE_START_TABLE = pd.concat([
    detail['starts'].assign(Policy=name) for name, detail in AUDIT_ADAPTIVE_DETAIL.items() if 'starts' in detail and not detail['starts'].empty
], ignore_index=True)
print('Online-MLE parameter paths, including ratios, slope z-score, selected position, and later block P&L:')
display(AUDIT_MLE_PARAMETER_TABLE.round(5))
print('Alternative-start convergence summary:')
display(AUDIT_MLE_START_TABLE.groupby(['Policy', 'Refit day']).agg(
    Starts=('Optimizer start', 'count'), Successful=('Success', 'sum'),
    Best_objective=('Objective', 'min'), Worst_objective=('Objective', 'max'),
    q_level_min=('q_level', 'min'), q_level_max=('q_level', 'max'),
    q_slope_min=('q_slope', 'min'), q_slope_max=('q_slope', 'max'),
    R_min=('R', 'min'), R_max=('R', 'max')
).reset_index().round(5))

# Genuine non-overlapping walk-forward blocks.  Selection at block start uses
# price[:start], and the selected fixed filter is frozen for the next block.
AUDIT_WALK_BOUNDARIES = [60, 120, 180, 240, 300, 364]
AUDIT_WALK_ROWS = []
fixed_state_paths = {
    label: audit_kalman_state(AUDIT_PRICE, parameters, gate=1.0)[0]
    for label, parameters in AUDIT_KALMAN_SPECS.items()
}
for start, end in zip(AUDIT_WALK_BOUNDARIES[:-1], AUDIT_WALK_BOUNDARIES[1:]):
    scores = {label: audit_score_window(AUDIT_PRICE, parameters, start, 'expanding') for label, parameters in AUDIT_KALMAN_SPECS.items()}
    chosen = max(('K2', 'K1', 'K3'), key=lambda label: (scores[label], {'K2': 2, 'K1': 1, 'K3': 0}[label]))
    chosen_position = audit_position_from_state(fixed_state_paths[chosen], -800)
    row = {'Test block': f'{start}:{end}', 'Chosen using price before block': chosen, 'K1 LL': scores['K1'], 'K2 LL': scores['K2'], 'K3 LL': scores['K3'], 'Selector test P&L': float(np.sum(realized_pnl(AUDIT_PRICE, chosen_position)[start:end]))}
    for label, position in [('Always-long', AUDIT_ALWAYS_LONG), ('Fixed K1', audit_position_from_state(fixed_state_paths['K1'], -800)), ('Fixed K2', audit_position_from_state(fixed_state_paths['K2'], -800))]:
        row[label + ' test P&L'] = float(np.sum(realized_pnl(AUDIT_PRICE, position)[start:end]))
    AUDIT_WALK_ROWS.append(row)
AUDIT_WALK_FORWARD = pd.DataFrame(AUDIT_WALK_ROWS)
print('Non-overlapping walk-forward comparison; delayed blocks are not independent samples.')
display(AUDIT_WALK_FORWARD.round(2))

# Delayed starts are reported only as overlapping initialization stress tests.
AUDIT_DELAYED_STARTS = [0, 30, 60, 90, 120, 180, 240]
AUDIT_DELAYED_ROWS = []
for name in AUDIT_SERIOUS_NAMES:
    position = AUDIT_ALWAYS_LONG if name == 'always_long' else AUDIT_CANDIDATES[name]
    for start in AUDIT_DELAYED_STARTS:
        AUDIT_DELAYED_ROWS.append({'Candidate': name, 'Pseudo-start': start, 'Suffix P&L': float(np.sum(realized_pnl(AUDIT_PRICE, position)[start:]))})
AUDIT_DELAYED = pd.DataFrame(AUDIT_DELAYED_ROWS)
print('Overlapping delayed-start suffixes are initialization stress tests, not independent validation samples.')
display(AUDIT_DELAYED.pivot(index='Candidate', columns='Pseudo-start', values='Suffix P&L').round(2))


def audit_transform_paths(price):
    price = np.asarray(price, dtype=float)
    changes = np.diff(price)
    paths = {
        'base': price.copy(),
        'price_units_x0p5': price * 0.5,
        'price_units_x1p5': price * 1.5,
        'price_units_x2': price * 2.0,
        'changes_x0p5': np.r_[price[0], price[0] + np.cumsum(changes * 0.5)],
        'changes_x1p5': np.r_[price[0], price[0] + np.cumsum(changes * 1.5)],
        'changes_x2': np.r_[price[0], price[0] + np.cumsum(changes * 2.0)],
        'volatility_shift_x2': np.r_[price[0], price[0] + np.cumsum(np.r_[changes[:len(changes) // 2], 2.0 * changes[len(changes) // 2:]])],
        'start_plus_50_same_changes': price + 50.0,
    }
    assert all(np.min(path) > 0 for path in paths.values())
    return paths


AUDIT_STRESS_NAMES = [
    'fixed_K1_c1_full_short', 'fixed_K2_c1_full_short', 'fixed_K3_c1_full_short',
    'K1_K2_consensus_full_short', 'K1_K2_K3_ensemble_full_short',
    'selector_w60_expanding', 'selector_w90_expanding',
    'online_mle_w60_expanding',
    'scale_log_K2', 'scale_ewma_0p9', 'scale_ewma_0p98',
]
AUDIT_STRESS_NAMES = [name for name in AUDIT_STRESS_NAMES if name in AUDIT_CANDIDATES]
base_signs = {name: np.sign(AUDIT_CANDIDATES[name]) for name in AUDIT_STRESS_NAMES}
AUDIT_SCALE_STRESS_ROWS = []
for transform_name, transformed_price in audit_transform_paths(AUDIT_PRICE).items():
    transformed_fixed, _ = build_audit_fixed_policies(transformed_price)
    transformed_adaptive, _, _ = build_audit_adaptive_policies(transformed_price, include_mle=True, mle_warmups=(60,), mle_scoring=('expanding',))
    transformed = dict(transformed_fixed)
    transformed.update(transformed_adaptive)
    for name in AUDIT_STRESS_NAMES:
        transformed_position = transformed[name]
        comparison = base_signs[name][20:] == np.sign(transformed_position[20:])
        AUDIT_SCALE_STRESS_ROWS.append({
            'Transform': transform_name, 'Candidate': name,
            'Position sign similarity after day 20': float(np.mean(comparison)),
            'Transformed P&L': float(np.sum(realized_pnl(transformed_price, transformed_position))),
            'Transformed max drawdown': float(max_drawdown_from_pnl(realized_pnl(transformed_price, transformed_position))),
        })
AUDIT_SCALE_STRESS = pd.DataFrame(AUDIT_SCALE_STRESS_ROWS)
print('Mechanical scale and starting-level stress tests. Similarity compares causal long/short/flat classifications after day 20 with Round 1.')
display(AUDIT_SCALE_STRESS.round(3))

# Strict causality audit for all fixed, selector, scale, and online-MLE policies.
AUDIT_CAUSALITY_DAYS = [0, 30, 120, 240, 300]
AUDIT_CAUSALITY_ROWS = []
for day in AUDIT_CAUSALITY_DAYS:
    altered = AUDIT_PRICE.copy()
    if day + 1 < AUDIT_N:
        altered[day + 1:] = altered[day] + 1000.0 * np.arange(1, AUDIT_N - day)
    altered_fixed, _ = build_audit_fixed_policies(altered)
    altered_adaptive, _, _ = build_audit_adaptive_policies(altered, include_mle=True)
    altered_all = dict(altered_fixed)
    altered_all.update(altered_adaptive)
    changed = []
    for name, position in AUDIT_CANDIDATES.items():
        if int(position[day]) != int(altered_all[name][day]):
            changed.append(name)
    AUDIT_CAUSALITY_ROWS.append({'Evaluation day': day, 'Policies tested': len(AUDIT_CANDIDATES), 'Changed positions': len(changed), 'Changed names': ', '.join(changed[:3])})
    assert not changed
AUDIT_CAUSALITY = pd.DataFrame(AUDIT_CAUSALITY_ROWS)
print('Causality proof: changing every observation after the audited day leaves that day position unchanged.')
display(AUDIT_CAUSALITY)


def audit_sample_circular_path(price, block_size, rng):
    price = np.asarray(price, dtype=float)
    changes = np.diff(price)
    sampled = []
    while len(sampled) < len(changes):
        start = int(rng.integers(0, len(changes)))
        indices = (start + np.arange(int(block_size))) % len(changes)
        sampled.extend(changes[indices].tolist())
    sampled = np.asarray(sampled[:len(changes)], dtype=float)
    path = np.r_[price[0], price[0] + np.cumsum(sampled)]
    # Keep log-price candidates defined.  This guard is only a path validity
    # safeguard and is counted rather than treated as a strategy decision.
    guard = int(np.min(path) <= 1.0)
    if guard:
        path = path + (1.0 - float(np.min(path)) + 1e-6)
    return path, guard


def build_audit_fast_policies(price):
    fixed, fixed_catalog = build_audit_fixed_policies(price)
    adaptive, adaptive_catalog, detail = build_audit_adaptive_policies(price, include_mle=False)
    combined = {'always_long': np.full(len(price), AUDIT_LIMIT, dtype=int)}
    combined.update(fixed)
    combined.update(adaptive)
    return combined


def build_audit_bootstrap_policies(price):
    """Build only the named serious fast bootstrap policies.

    The full robustness surface remains in AUDIT_TABLE and the family
    catalogue.  It is not recomputed on every bootstrap path because that
    would spend runtime on redundant diagnostics rather than validation of
    the serious policy choices.
    """
    price = np.asarray(price, dtype=float)
    states = {label: audit_kalman_state(price, parameters, gate=1.0)[0] for label, parameters in AUDIT_KALMAN_SPECS.items()}
    source = {'always_long': np.full(len(price), AUDIT_LIMIT, dtype=int)}
    for label in ('K1', 'K2', 'K3'):
        source[f'fixed_{label}_c1_full_short'] = audit_position_from_state(states[label], -800)
    source['K1_K2_consensus_full_short'] = np.where((states['K1'] < 0) & (states['K2'] < 0), -800, AUDIT_LIMIT).astype(int)
    source['K1_K2_K3_ensemble_full_short'] = np.where(((states['K1'] < 0).astype(int) + (states['K2'] < 0).astype(int) + (states['K3'] < 0).astype(int)) >= 2, -800, AUDIT_LIMIT).astype(int)
    source['K1_K2_consensus_full_short'][0] = 0
    source['K1_K2_K3_ensemble_full_short'][0] = 0
    adaptive, _, _ = build_audit_adaptive_policies(price, include_mle=False)
    source.update(adaptive)
    return {name: source[name] for name in AUDIT_BOOTSTRAP_FAST_NAMES}


# Correct price-path bootstrap.  The earlier notebook section has the larger
# 2,000-repetition old-catalogue validation.  This focused extension uses a
# runtime-bounded count for the new adaptive policies; every draw still
# rebuilds prices and reruns positions rather than resampling fixed P&L.
AUDIT_BOOTSTRAP_FAST_NAMES = [
    'always_long', 'fixed_K1_c1_full_short', 'fixed_K2_c1_full_short', 'fixed_K3_c1_full_short',
    'K1_K2_consensus_full_short', 'K1_K2_K3_ensemble_full_short',
    'selector_w60_expanding', 'selector_w90_expanding', 'selector_w60_trailing60', 'selector_w90_trailing120',
    'scale_log_K2', 'scale_ewma_0p9', 'scale_ewma_0p98',
]
AUDIT_BOOTSTRAP_FAST_NAMES = [name for name in AUDIT_BOOTSTRAP_FAST_NAMES if name == 'always_long' or name in AUDIT_CANDIDATES]
AUDIT_BOOTSTRAP_BLOCKS = (5, 10, 20)
AUDIT_BOOTSTRAP_FAST_REPETITIONS = 100
AUDIT_BOOTSTRAP_MLE_REPETITIONS = 10
AUDIT_BOOTSTRAP_RNG = np.random.default_rng(20260811)
AUDIT_BOOTSTRAP_FAST_SUMMARIES = []
AUDIT_BOOTSTRAP_FAST_GUARDS = []
AUDIT_BOOTSTRAP_FAST_RAW = {}
for block_size in AUDIT_BOOTSTRAP_BLOCKS:
    totals = np.empty((AUDIT_BOOTSTRAP_FAST_REPETITIONS, len(AUDIT_BOOTSTRAP_FAST_NAMES)), dtype=float)
    always_totals = np.empty(AUDIT_BOOTSTRAP_FAST_REPETITIONS, dtype=float)
    guards = 0
    for repetition in range(AUDIT_BOOTSTRAP_FAST_REPETITIONS):
        synthetic_price, guard = audit_sample_circular_path(AUDIT_PRICE, block_size, AUDIT_BOOTSTRAP_RNG)
        guards += guard
        synthetic = build_audit_bootstrap_policies(synthetic_price)
        always_total = float(np.sum(realized_pnl(synthetic_price, synthetic['always_long'])))
        always_totals[repetition] = always_total
        for column, name in enumerate(AUDIT_BOOTSTRAP_FAST_NAMES):
            totals[repetition, column] = float(np.sum(realized_pnl(synthetic_price, synthetic[name])))
    AUDIT_BOOTSTRAP_FAST_RAW[block_size] = (totals, always_totals)
    AUDIT_BOOTSTRAP_FAST_GUARDS.append({'Block size': block_size, 'Repetitions': AUDIT_BOOTSTRAP_FAST_REPETITIONS, 'Positive-price guards': guards})
    for column, name in enumerate(AUDIT_BOOTSTRAP_FAST_NAMES):
        values = totals[:, column]
        incremental = values - always_totals
        AUDIT_BOOTSTRAP_FAST_SUMMARIES.append({
            'Block size': block_size, 'Candidate': name,
            'P(P&L>0)': float(np.mean(values > 0)),
            'P(beats always-long)': float(np.mean(incremental > 0)),
            'Median P&L': float(np.median(values)),
            'Median incremental': float(np.median(incremental)),
            'P5 P&L': float(np.quantile(values, 0.05)),
            'P5 incremental': float(np.quantile(incremental, 0.05)),
            'P95 P&L': float(np.quantile(values, 0.95)),
        })
AUDIT_BOOTSTRAP_FAST_SUMMARY = pd.DataFrame(AUDIT_BOOTSTRAP_FAST_SUMMARIES)
print(f'Corrected price-path bootstrap for fixed/selector/scale policies: {AUDIT_BOOTSTRAP_FAST_REPETITIONS} causal reruns per circular block size; the earlier section contains the larger old-catalogue run.')
display(pd.DataFrame(AUDIT_BOOTSTRAP_FAST_GUARDS))
display(AUDIT_BOOTSTRAP_FAST_SUMMARY.round(2))

# MLE bootstrap is kept separate and explicitly labelled because it is much
# more expensive than rerunning a fixed policy.  It still performs a causal
# refit on every reconstructed path and reports the same quantities.
AUDIT_BOOTSTRAP_MLE_NAMES = [name for name in ('online_mle_w60_expanding', 'online_mle_w60_trailing120', 'online_mle_w90_expanding', 'online_mle_w90_trailing120') if name in AUDIT_CANDIDATES]
AUDIT_BOOTSTRAP_MLE_SUMMARIES = []
AUDIT_BOOTSTRAP_MLE_GUARDS = []
AUDIT_BOOTSTRAP_MLE_RAW = {}
for block_size in AUDIT_BOOTSTRAP_BLOCKS:
    totals = np.empty((AUDIT_BOOTSTRAP_MLE_REPETITIONS, len(AUDIT_BOOTSTRAP_MLE_NAMES)), dtype=float)
    always_totals = np.empty(AUDIT_BOOTSTRAP_MLE_REPETITIONS, dtype=float)
    guards = 0
    for repetition in range(AUDIT_BOOTSTRAP_MLE_REPETITIONS):
        synthetic_price, guard = audit_sample_circular_path(AUDIT_PRICE, block_size, AUDIT_BOOTSTRAP_RNG)
        guards += guard
        synthetic_fixed = build_audit_bootstrap_policies(synthetic_price)
        synthetic_adaptive, _, _ = build_audit_adaptive_policies(synthetic_price, include_mle=True)
        always_total = float(np.sum(realized_pnl(synthetic_price, synthetic_fixed['always_long'])))
        always_totals[repetition] = always_total
        for column, name in enumerate(AUDIT_BOOTSTRAP_MLE_NAMES):
            totals[repetition, column] = float(np.sum(realized_pnl(synthetic_price, synthetic_adaptive[name])))
    AUDIT_BOOTSTRAP_MLE_RAW[block_size] = (totals, always_totals)
    AUDIT_BOOTSTRAP_MLE_GUARDS.append({'Block size': block_size, 'Repetitions': AUDIT_BOOTSTRAP_MLE_REPETITIONS, 'Positive-price guards': guards})
    for column, name in enumerate(AUDIT_BOOTSTRAP_MLE_NAMES):
        values = totals[:, column]
        incremental = values - always_totals
        AUDIT_BOOTSTRAP_MLE_SUMMARIES.append({
            'Block size': block_size, 'Candidate': name,
            'P(P&L>0)': float(np.mean(values > 0)), 'P(beats always-long)': float(np.mean(incremental > 0)),
            'Median P&L': float(np.median(values)), 'Median incremental': float(np.median(incremental)),
            'P5 P&L': float(np.quantile(values, 0.05)), 'P5 incremental': float(np.quantile(incremental, 0.05)),
            'P95 P&L': float(np.quantile(values, 0.95)),
        })
AUDIT_BOOTSTRAP_MLE_SUMMARY = pd.DataFrame(AUDIT_BOOTSTRAP_MLE_SUMMARIES)
print(f'Corrected price-path bootstrap for online MLE: {AUDIT_BOOTSTRAP_MLE_REPETITIONS} causal reruns per block size; lower count is a runtime limitation, not fixed-P&L resampling.')
display(pd.DataFrame(AUDIT_BOOTSTRAP_MLE_GUARDS))
display(AUDIT_BOOTSTRAP_MLE_SUMMARY.round(2))

# The prior section already contains a 2,000-repetition null over the old
# combined catalogue.  This extension therefore tests the new serious fast
# policies as a separate family, while the online-MLE family remains separate
# because its per-path optimization is much more expensive.
AUDIT_FAST_CATALOG_NAMES = list(AUDIT_BOOTSTRAP_FAST_NAMES)
AUDIT_FAST_OBSERVED_TOTALS = {name: float(np.sum(realized_pnl(AUDIT_PRICE, position))) for name, position in build_audit_bootstrap_policies(AUDIT_PRICE).items()}
AUDIT_FAST_OBSERVED_BEST_NAME = max(AUDIT_FAST_OBSERVED_TOTALS, key=AUDIT_FAST_OBSERVED_TOTALS.get)
AUDIT_FAST_OBSERVED_BEST = AUDIT_FAST_OBSERVED_TOTALS[AUDIT_FAST_OBSERVED_BEST_NAME]
AUDIT_FAMILYWISE_REPETITIONS = 100
AUDIT_FAMILYWISE_BLOCK_SIZE = 10
AUDIT_FAMILYWISE_RNG = np.random.default_rng(20260812)
AUDIT_FAMILYWISE_BLOCK_MAX = np.empty(AUDIT_FAMILYWISE_REPETITIONS, dtype=float)
AUDIT_FAMILYWISE_PERMUTED_MAX = np.empty(AUDIT_FAMILYWISE_REPETITIONS, dtype=float)
family_guards = 0
for repetition in range(AUDIT_FAMILYWISE_REPETITIONS):
    block_price, guard = audit_sample_circular_path(AUDIT_PRICE, AUDIT_FAMILYWISE_BLOCK_SIZE, AUDIT_FAMILYWISE_RNG)
    family_guards += guard
    fast_new = build_audit_bootstrap_policies(block_price)
    AUDIT_FAMILYWISE_BLOCK_MAX[repetition] = max(float(np.sum(realized_pnl(block_price, position))) for position in fast_new.values())
    shuffled = AUDIT_FAMILYWISE_RNG.permutation(np.diff(AUDIT_PRICE))
    permuted_price = np.r_[AUDIT_PRICE[0], AUDIT_PRICE[0] + np.cumsum(shuffled)]
    if np.min(permuted_price) <= 1.0:
        permuted_price = permuted_price + (1.0 - float(np.min(permuted_price)) + 1e-6)
    perm_new = build_audit_bootstrap_policies(permuted_price)
    AUDIT_FAMILYWISE_PERMUTED_MAX[repetition] = max(float(np.sum(realized_pnl(permuted_price, position))) for position in perm_new.values())


def audit_mc_summary(values, observed, label):
    hits = int(np.sum(values >= observed))
    p_value = (1.0 + hits) / (len(values) + 1.0)
    se = math.sqrt(p_value * (1.0 - p_value) / len(values))
    return {
        'Null': label, 'New serious fast candidates': len(AUDIT_FAST_CATALOG_NAMES),
        'Observed best candidate': AUDIT_FAST_OBSERVED_BEST_NAME, 'Observed best P&L': AUDIT_FAST_OBSERVED_BEST,
        'Repetitions': len(values), 'Null median': float(np.median(values)), 'Null 95%': float(np.quantile(values, 0.95)),
        'Exceedances': hits, 'Monte Carlo p-value': p_value, 'Monte Carlo SE': se,
        'Approx MC 95% low': max(0.0, p_value - 1.96 * se), 'Approx MC 95% high': min(1.0, p_value + 1.96 * se),
    }


AUDIT_FAMILYWISE = pd.DataFrame([
    audit_mc_summary(AUDIT_FAMILYWISE_BLOCK_MAX, AUDIT_FAST_OBSERVED_BEST, 'Circular blocks length 10'),
    audit_mc_summary(AUDIT_FAMILYWISE_PERMUTED_MAX, AUDIT_FAST_OBSERVED_BEST, 'Daily-change permutation'),
])
AUDIT_FAMILYWISE['Positive-price guards in block null'] = family_guards
print(f'Family-wise null over the new serious fast candidates; {AUDIT_FAMILYWISE_REPETITIONS} maxima per null in this runtime-bounded extension. The earlier section contains the 2,000-repetition legacy combined null.')
display(AUDIT_FAMILYWISE.round(4))
print('The online-MLE candidates are not hidden in this maximum: their path bootstrap and parameter instability are reported separately, and their Round 1 maxima are compared with the fixed family.')

# A compact long/short attribution table for the serious policies.
AUDIT_ATTRIBUTION = AUDIT_TABLE.loc[AUDIT_SERIOUS_NAMES, ['P&L', 'Incremental vs always long', 'Long P&L', 'Short P&L', 'Long days', 'Short days', 'Flat days', 'Negative K2 regimes']].copy()
display(AUDIT_ATTRIBUTION.round(2))

print('Audit integrity checks:')
for name, position in [('always_long', AUDIT_ALWAYS_LONG), *list(AUDIT_CANDIDATES.items())]:
    assert np.issubdtype(np.asarray(position).dtype, np.integer)
    assert np.max(np.abs(position)) <= AUDIT_LIMIT
    assert np.max(np.abs(position * AUDIT_PRICE)) <= JEANS_BUDGET
print('All focused policies pass integer, ±800 Jeans-limit, and Jeans-only gross-value checks.')
