"""
liferaft_strategy.py
=====================
A self-contained strategy for the AlgoJam "Liferaft Ticket" instrument.

Liferaft is a minority game, not a price-history instrument:
  - Position limit is +/- 1 ticket. PnL each day = position * (price change).
  - Each day the price moves based on the CROWD (all teams that take a side):
        majority LONG  -> price FALLS $5,000   (so longs lose, shorts win)
        majority SHORT -> price RISES $8,000   (so shorts lose, longs win)
        tie / all flat -> no change
  - Price cannot fall below $20,000. Starts at $100,000.
  - You cannot backtest it on Round 1 data (the sim forces its PnL to 0),
    so this is a *live rule* that reads the revealed price to infer the crowd.

You can reverse-engineer what the crowd did from each day's price move:
        a -$5,000 day  =>  the majority went LONG that day
        a +$8,000 day  =>  the majority went SHORT that day
        no change      =>  tie, everyone flat, OR a long-majority clamped at the floor

Policy implemented here ("prior-guided cycle hunter + asymmetric fader"):
  1. Round-1 prior: the competition provides a full prior year of Liferaft
     prices. That tape shows a field long on 58.8% of deciding days -- BELOW
     the 8/13 ~= 61.5% breakeven -- so the price drifted UP ~$356/day
     (+$130k over the year: always-long was profitable), never came near
     the floor (min $75k), and contained no detectable cycles. These
     measurements enter as a WEAK Bayesian prior (pseudo-counts), so the
     algorithm is never flying blind on early Round-2 days, while ~2 weeks
     of live evidence outweighs the prior if the real crowd differs.
  2. Hunt cycles: every competitor is a deterministic script reacting to the
     same public price history, and deterministic feedback systems fall into
     repeating loops (e.g. everyone flips sides after a losing day -> a 2-day
     L/S oscillation). Watch a rolling observation window; if the crowd
     sequence repeats with a stable period, tomorrow's majority is known in
     advance -- take the minority side and collect every day the cycle holds.
     The short-side safety gates still apply.
  3. Fade a persistent crowd bias, ASYMMETRICALLY:
       - LONG on moderate evidence of a short-leaning crowd. Each continuation
         day pays +$8,000, the upside has no ceiling, and a wrong long only
         costs $5,000.
       - SHORT only on stronger evidence of a long-leaning crowd, only while
         price is comfortably above the floor, AND only when the longer-run
         record shows the crowd long on well over 8/13 ~= 62% of deciding
         days -- the exact breakeven (set by the 5:8 payoff ratio) past which
         shorting beats staying flat. Near that line the game is unexploitable
         and the right trade is no trade.
  4. Near the $20,000 floor the game freezes: shorting there can only lose,
     so no rational team does it, the majority stays long, and the clamp eats
     every move. Default is FLAT (frees budget). Set FLOOR_ACTION = 1 if you
     would rather hold the never-losing long that cashes +$8,000 on any
     irrational short-majority day.
  5. Drift default: with no cycle and no streak on, hold LONG while the
     prior-blended crowd long-share stays clearly below breakeven (in that
     regime a held long is +EV every single day); go FLAT only when the
     estimate approaches breakeven, where no side has an edge.

The decision depends only on the price history, never on self.day, so it
behaves identically whether Round 2 supplies fresh data or Round 1 prepended.

------------------------------------------------------------------------------
HOW TO WIRE IT IN (you do this; algorithm.py is untouched by this file):

    from liferaft_strategy import liferaft_position   # add near the top

    # ...inside get_positions(), after you build desiredPositions:
    if "Liferaft Ticket" in self.positionLimits:
        desiredPositions["Liferaft Ticket"] = liferaft_position(
            self.data["Liferaft Ticket"],   # price history up to & incl. today
            self.day,
        )

That's it. The function always returns an int in {-1, 0, 1}.
------------------------------------------------------------------------------
"""

# ---- Tunable constants ------------------------------------------------------
FLOOR              = 20_000   # hard price floor from the rules
FLOOR_ZONE         = 25_000   # treat price <= this as "near the floor" -> FLOOR_ACTION
LONG_MAJ_MOVE      = -5_000   # price change that reveals a long-majority day
SHORT_MAJ_MOVE     =  8_000   # price change that reveals a short-majority day
MOVE_TOL           =  1       # float tolerance when classifying a move
LOOKBACK           =  5       # how many recent readable days to weigh
BIAS_MIN_LONG      =  2       # net crowd-short lead needed to go LONG (fade pays +$8k)
BIAS_MIN_SHORT     =  3       # net crowd-long lead needed to go SHORT (fade pays +$5k)
                              # Asymmetric on purpose: a wrong short costs $8k vs a
                              # wrong long's $5k, and against pure noise long-fades
                              # are +EV while short-fades are -EV, so shorting
                              # demands stronger evidence.
SHORT_MIN_PRICE    = 30_000   # never OPEN a short below this. A short's maximum
                              # remaining profit is (price - $20k floor); demand it
                              # at least exceed one adverse +$8k move.
REGIME_WINDOW      = 30       # readable days used to estimate the crowd's long-share
SHORT_REGIME_MIN   = 0.65     # allow shorts only if that share clears the 8/13
                              # ~= 62% breakeven (with margin). Below it, shorting
                              # can't beat staying flat even when timed well.
CYCLE_WINDOW       = 24       # observation range scanned for repeating patterns
CYCLE_MAX_PERIOD   = 8        # longest cycle length considered (window/3)
CYCLE_MATCH_MIN    = 0.85     # fraction of the window that must repeat exactly
                              # before a cycle is trusted. High enough that iid
                              # noise almost never triggers it by chance.
CYCLE_MIN_COMPS    = 10       # minimum day-vs-day comparisons required
PRIOR_LONG_SHARE   = 0.588    # measured off the provided Round-1 tape
                              # (214 long-majority days / 364 deciding days)
PRIOR_WEIGHT       = 15       # pseudo-days of belief in that prior. Small on
                              # purpose: ~2 weeks of live Round-2 data outweighs
                              # it if the real crowd behaves differently.
DRIFT_LONG_MAX     = 0.60     # hold the drift-capture long while the blended
                              # long-share estimate is at or below this
                              # (safety margin under the 8/13 ~= 0.615 breakeven)
FLOOR_ACTION       =  0       # position to hold inside the floor zone.
                              # 0 = flat (default): the market rationally freezes at
                              #     the floor, so a position earns ~$0/day and only
                              #     consumes budget.
                              # 1 = hold the "free option" long that can't lose at
                              #     the exact floor and pays +$8k if the room ever
                              #     irrationally flips short.
# -----------------------------------------------------------------------------


def _classify_move(diff):
    """Turn one day's price change into a crowd signal.
    Returns 'L' (majority long), 'S' (majority short), or None (no signal)."""
    if abs(diff - LONG_MAJ_MOVE) <= MOVE_TOL:
        return "L"
    if abs(diff - SHORT_MAJ_MOVE) <= MOVE_TOL:
        return "S"
    return None


def _calendar_signals(prices):
    """One symbol per calendar day: 'L', 'S', or '0' (tie/all-flat/clamped).
    Calendar alignment matters for cycle detection, so unreadable days are
    kept as '0' rather than dropped."""
    out = []
    for i in range(1, len(prices)):
        s = _classify_move(prices[i] - prices[i - 1])
        out.append(s if s is not None else "0")
    return out


def _detect_cycle(seq):
    """Scan the recent observation window for a repeating crowd pattern.

    For each candidate period p, check how much of the window satisfies
    seq[i] == seq[i-p]. If a high enough fraction repeats, the field has
    locked into a loop and tomorrow's symbol is seq[-p].

    A constant run (all 'L') technically matches every period but is a trend,
    not a cycle -- the bias-fader owns that case (with its safety gates), so
    the repeating motif must contain both sides.

    Returns (period, predicted_next_symbol) or None.
    """
    w = seq[-CYCLE_WINDOW:]
    for p in range(2, CYCLE_MAX_PERIOD + 1):
        n = len(w) - p
        if n < max(CYCLE_MIN_COMPS, 2 * p):
            continue
        matches = sum(w[i] == w[i - p] for i in range(p, len(w)))
        if matches / n < CYCLE_MATCH_MIN:
            continue
        motif = w[-p:]
        if "L" in motif and "S" in motif:
            return p, w[-p]
    return None


def liferaft_position(prices, day=None):
    """
    Decide today's Liferaft position.

    prices : list of the Liferaft price for every day up to and including today
             (this is exactly what self.data["Liferaft Ticket"] gives you).
    day    : accepted for backwards compatibility but unused -- the decision
             depends only on the price history.

    Returns an int in {-1, 0, 1}.
    """
    if not prices:
        return 0

    price_today = prices[-1]

    # 1) Read the whole available history into calendar symbols. No warm-up
    #    period: the Round-1 prior carries the early days instead.
    seq = _calendar_signals(prices)

    # 2) Observe-and-capitalize: if the other bots have locked into a
    #    repeating cycle, tomorrow's majority is predictable -- take the
    #    minority side. Sharper than the statistical fader, so it runs first.
    cycle = _detect_cycle(seq)
    if cycle is not None:
        _, predicted = cycle
        if predicted == "S":
            return 1    # crowd about to sell -> be long for the +$8k day
        if predicted == "L" and price_today >= SHORT_MIN_PRICE:
            return -1   # crowd about to buy -> be short for the -$5k day
        # Predicted '0', or an 'L' day too close to the floor to be worth
        # shorting: fall through to the normal gates below.

    # 3) Floor zone: the game freezes here (no rational team shorts at the
    #    floor, so the majority stays long and the clamp eats every move).
    #    Default FLOOR_ACTION = 0: go flat, free the budget.
    if price_today <= FLOOR_ZONE:
        return FLOOR_ACTION

    # 4) Read the crowd from recent readable days.
    signals = [s for s in seq if s != "0"]
    recent = signals[-LOOKBACK:]
    longs  = recent.count("L")
    shorts = recent.count("S")
    net    = longs - shorts   # positive => crowd leans long

    # 5) Regime estimate, blended with the Round-1 prior via pseudo-counts.
    #    With no live data yet this equals PRIOR_LONG_SHARE; after a few weeks
    #    the live window dominates. Shorting only beats flat past the 8/13
    #    breakeven, so below SHORT_REGIME_MIN shorts stay switched off.
    regime = signals[-REGIME_WINDOW:]
    long_share = ((regime.count("L") + PRIOR_LONG_SHARE * PRIOR_WEIGHT)
                  / (len(regime) + PRIOR_WEIGHT))
    shorts_allowed = long_share >= SHORT_REGIME_MIN

    # 6) Fade a clear, persistent bias.
    #    Shorts need stronger evidence, a hot-enough regime, AND enough room
    #    left above the floor to be worth one adverse move; longs face no
    #    ceiling and cheaper mistakes, so the bar is lower.
    if net >= BIAS_MIN_SHORT and price_today >= SHORT_MIN_PRICE and shorts_allowed:
        return -1   # crowd keeps buying -> price keeps falling -> short it
    if -net >= BIAS_MIN_LONG:
        return 1    # crowd keeps selling -> price keeps rising -> long it

    # 7) Drift default. Round 1 measured the field long on just 58.8% of
    #    deciding days -- below breakeven -- which makes simply holding long
    #    +EV (+$356/day on that tape). Capture that drift while the blended
    #    estimate stays clearly below breakeven; otherwise stand flat.
    if long_share <= DRIFT_LONG_MAX:
        return 1
    return 0


# =============================================================================
# Built-in self-test / demo.
# You can't backtest Liferaft on the real data, so this fabricates a crowd so
# you can watch the rule react. Run:  python liferaft_strategy.py
# =============================================================================
if __name__ == "__main__":
    import random

    def simulate(crowd_long_prob=0.5, pattern=None, noise=0.0,
                 days=365, seed=0):
        """Fabricate a crowd and score our rule against it.

        pattern : e.g. "LS" or "LLS" -- the field repeats this cycle forever,
                  with each day's side flipped with probability `noise`
                  (models mostly-deterministic bots with occasional deviations).
        If pattern is None, each day's majority is long with probability
        crowd_long_prob, independently (a statistical, non-cyclic field).
        """
        rng = random.Random(seed)
        prices = [100_000]
        pnl = 0
        for d in range(days):
            # Our rule decides a position using history up to & including today.
            pos_today = liferaft_position(prices, d)

            # The crowd acts and the price moves.
            if pattern is not None:
                crowd = pattern[d % len(pattern)]
                if rng.random() < noise:
                    crowd = "S" if crowd == "L" else "L"
            else:
                crowd = "L" if rng.random() < crowd_long_prob else "S"
            if crowd == "L":
                new_price = max(FLOOR, prices[-1] + LONG_MAJ_MOVE)
            else:
                new_price = prices[-1] + SHORT_MAJ_MOVE

            # We earn on the move from today's price to tomorrow's.
            pnl += pos_today * (new_price - prices[-1])
            prices.append(new_price)
        return pnl

    print("=" * 68)
    print("Liferaft strategy self-test (synthetic crowds)")
    print("=" * 68)

    scenarios = [
        ("Long-biased field (60% long)",       dict(crowd_long_prob=0.60)),
        ("Heavily long-biased (75% long)",     dict(crowd_long_prob=0.75)),
        ("Short-biased field (35% long)",      dict(crowd_long_prob=0.35)),
        ("Balanced field (50/50)",             dict(crowd_long_prob=0.50)),
        ("2-day cycle L,S (deterministic)",    dict(pattern="LS")),
        ("3-day cycle L,L,S (deterministic)",  dict(pattern="LLS")),
        ("2-day cycle, 5% noise",              dict(pattern="LS", noise=0.05)),
        ("2-day cycle, 10% noise",             dict(pattern="LS", noise=0.10)),
    ]
    for name, kw in scenarios:
        # average over several seeds to smooth out luck
        results = [simulate(seed=s, **kw) for s in range(200)]
        avg = sum(results) / len(results)
        print(f"{name:36s}  avg PnL over 200 runs: ${avg:>12,.0f}")
