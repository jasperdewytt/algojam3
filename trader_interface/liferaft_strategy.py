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

Policy implemented here ("asymmetric fader with floor exit"):
  1. Warm up: stay FLAT for the first few days (no signal yet).
  2. Fade a persistent crowd bias, ASYMMETRICALLY:
       - LONG on moderate evidence of a short-leaning crowd. Each continuation
         day pays +$8,000, the upside has no ceiling, and a wrong long only
         costs $5,000.
       - SHORT only on stronger evidence of a long-leaning crowd, only while
         price is comfortably above the floor, AND only when the longer-run
         record shows the crowd long on well over 8/13 ~= 62% of deciding
         days -- the exact breakeven (set by the 5:8 payoff ratio) past which
         shorting beats staying flat. Near that line the game is unexploitable
         and the right trade is no trade.
  3. Near the $20,000 floor the game freezes: shorting there can only lose,
     so no rational team does it, the majority stays long, and the clamp eats
     every move. Default is FLAT (frees budget). Set FLOOR_ACTION = 1 if you
     would rather hold the never-losing long that cashes +$8,000 on any
     irrational short-majority day.
  4. No clear signal -> FLAT (zero EV, zero risk) rather than a coin-flip.

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
WARMUP_DAYS        =  3       # stay flat until we have this many days of history
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


def liferaft_position(prices, day=None):
    """
    Decide today's Liferaft position.

    prices : list of the Liferaft price for every day up to and including today
             (this is exactly what self.data["Liferaft Ticket"] gives you).
    day    : optional; the current day index. If omitted it's inferred from len.

    Returns an int in {-1, 0, 1}.
    """
    if not prices:
        return 0
    if day is None:
        day = len(prices) - 1

    price_today = prices[-1]

    # 1) Warm-up: no readable history yet.
    if day < WARMUP_DAYS:
        return 0

    # 2) Floor zone: the game freezes here (no rational team shorts at the
    #    floor, so the majority stays long and the clamp eats every move).
    #    Default FLOOR_ACTION = 0: go flat, free the budget.
    if price_today <= FLOOR_ZONE:
        return FLOOR_ACTION

    # 3) Read the crowd from recent readable days.
    signals = []
    for i in range(1, len(prices)):
        s = _classify_move(prices[i] - prices[i - 1])
        if s is not None:
            signals.append(s)

    recent = signals[-LOOKBACK:]
    if not recent:
        return 0  # nothing readable (all ties/flat) -> stay out

    longs  = recent.count("L")
    shorts = recent.count("S")
    net    = longs - shorts   # positive => crowd leans long

    # 4) Regime gate for shorts: estimate the crowd's long-share over a longer
    #    window. Shorting only beats flat past the 8/13 breakeven, so below
    #    SHORT_REGIME_MIN the short side is switched off entirely.
    regime = signals[-REGIME_WINDOW:]
    long_share = regime.count("L") / len(regime)
    shorts_allowed = long_share >= SHORT_REGIME_MIN

    # 5) Fade a clear, persistent bias; otherwise stay flat.
    #    Shorts need stronger evidence, a hot-enough regime, AND enough room
    #    left above the floor to be worth one adverse move; longs face no
    #    ceiling and cheaper mistakes, so the bar is lower.
    if net >= BIAS_MIN_SHORT and price_today >= SHORT_MIN_PRICE and shorts_allowed:
        return -1   # crowd keeps buying -> price keeps falling -> short it
    if -net >= BIAS_MIN_LONG:
        return 1    # crowd keeps selling -> price keeps rising -> long it
    return 0


# =============================================================================
# Built-in self-test / demo.
# You can't backtest Liferaft on the real data, so this fabricates a crowd so
# you can watch the rule react. Run:  python liferaft_strategy.py
# =============================================================================
if __name__ == "__main__":
    import random

    def simulate(crowd_long_prob, days=365, seed=0, verbose=False):
        """Simulate a field whose majority is LONG with probability
        `crowd_long_prob` each day, apply the price rules, and score our rule."""
        rng = random.Random(seed)
        prices = [100_000]
        pnl = 0
        pos_today = 0
        for d in range(days):
            # Our rule decides a position using history up to & including today.
            pos_today = liferaft_position(prices, d)

            # The crowd acts and the price moves.
            crowd = "L" if rng.random() < crowd_long_prob else "S"
            if crowd == "L":
                new_price = max(FLOOR, prices[-1] + LONG_MAJ_MOVE)
            else:
                new_price = prices[-1] + SHORT_MAJ_MOVE

            # We earn on the move from today's price to tomorrow's.
            pnl += pos_today * (new_price - prices[-1])
            if verbose and d < 20:
                print(f"day {d:3d}  price ${prices[-1]:>8,}  ourpos {pos_today:+d}  "
                      f"crowd {crowd}  -> ${new_price:>8,}  cumPnL ${pnl:>10,}")
            prices.append(new_price)
        return pnl, prices[-1]

    print("=" * 64)
    print("Liferaft strategy self-test (synthetic crowds)")
    print("=" * 64)

    scenarios = [
        ("Long-biased field (60% long)",  0.60),
        ("Heavily long-biased (75% long)", 0.75),
        ("Short-biased field (35% long)", 0.35),
        ("Balanced field (50/50)",        0.50),
    ]
    for name, p in scenarios:
        # average over several seeds to smooth out luck
        results = [simulate(p, seed=s)[0] for s in range(200)]
        avg = sum(results) / len(results)
        print(f"{name:34s}  avg PnL over 200 runs: ${avg:>12,.0f}")

    print("-" * 64)
    print("Sample trace (75% long-biased field):")
    simulate(0.75, seed=1, verbose=True)
