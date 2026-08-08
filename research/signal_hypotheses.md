# Signal hypotheses from the Instrument Specification

Derived **only** from the narrative in `docs/AlgoJam3_Instrument_Specification.pdf`,
the event info PDF and `README.md`. No price data was inspected while writing this,
so every claim below is a *prior* to be tested, not a finding.

---

## 0. Structural facts that shape everything

**Scoring:** total profit over Round 2's 365 days; Sharpe is the tie-break. So
reliability is worth real points, not just comfort.

**Budget:** `Σ |position × price| ≤ 600,000` per day, and a violation *zeroes your
entire portfolio for that day* — it is not clipped, it is wiped. This makes the
budget check a hard safety rail, not an optimisation nicety.

**Capacity table** (max notional at each instrument's limit, using approximate
Round-1 end-of-year prices read off the spec charts):

| Instrument | Limit | ~Price | Max notional | % of 600k |
|---|---:|---:|---:|---:|
| MenuDash | 75,000 | 1.97 | 147,750 | 24.6% |
| Sausage Sizzle | 3,000 | 39.8 | 119,400 | 19.9% |
| Liferaft Ticket | 1 | 100,000 | 100,000 | 16.7% |
| Thrifted Jeans | 800 | 87 | 69,600 | 11.6% |
| UQ Dollar | 650 | 100 | 65,000 | 10.8% |
| Bread | 500 | 127 | 63,500 | 10.6% |
| Fintech Token | 100 | 530 | 53,000 | 8.8% |
| Boat Party Ticket | 1,000 | 45 | 45,000 | 7.5% |
| Sausage | 5,000 | 6.15 | 30,750 | 5.1% |
| **Total** | | | **694,000** | **116%** |

Everything at max is ~16% over budget. The constraint binds, but only mildly —
you must decline roughly one mid-sized instrument's worth of gross exposure.
**Bread is the obvious donor**: it eats 10.6% of the budget for what looks like the
smallest standalone edge in the set, and its real value is as a *model input*, not
a position.

**Timing convention:** `self.data[inst]` runs through today, and P&L accrues on
`position_t × (P_{t+1} − P_t)`. So any relationship of the form "instrument X today
is a function of instrument Y *yesterday*" is a one-day-ahead forecast with zero
look-ahead. Exactly one instrument is written that way (Sausage Sizzle) and it is
almost certainly the headline edge. **Confirm the accounting convention in
`simulation.py` before relying on this.**

---

## 1. Sausage Sizzle — the lagged cost identity

> "The price is really just the ingredients added up: bread, sausages, and paying
> whoever is on the tongs for four hours. Also, whoever is on the roster does the
> books at the end of the day rather than the start, so **today's price comes off
> yesterday's shopping**. Two of those three costs (bread and sausages) are tradable
> on the exchange; but, the third one is the volunteer — who, as it happens, also
> drives for MenuDash on weekends."

This is the most explicit hint in the document and it hands over a near-deterministic
model:

```
Sizzle_t = a·Bread_{t-1} + b·Sausage_{t-1} + c·L_{t-1} + const
```

where `L` is the latent hourly labour rate. Note "**four hours**" — the labour
coefficient should come out at roughly 4 × an hourly wage, which is a useful sanity
check on a fitted `c`.

### Why this is the biggest edge

At the close of day `t` you already know `Bread_t` and `Sausage_t` exactly. Bread's
own description says its weekly moves are "larger, **less predictable** jumps" —
i.e. unpredictable *for bread*, but fully known one day ahead *for the sizzle*. The
spec makes bread and sausage hard to trade outright and simultaneously makes them a
perfect leading indicator for something with 4× bread's position notional.

In differences:

```
ΔSizzle_{t+1} = a·ΔBread_t + b·ΔSausage_t + c·ΔL_t
```

The first two terms are known exactly. `ΔL_t` is small (see MenuDash below — the
latent cost is smooth). So the *sign* of tomorrow's sizzle move is dominated by
quantities you can already see.

### Recovering the latent labour rate exactly

Rearranging the identity for the current day:

```
c·L_{t-1} = Sizzle_t − a·Bread_{t-1} − b·Sausage_{t-1} − const
```

**The Sizzle is a noiseless read of yesterday's true labour cost.** MenuDash is a
*noisy* read of the same thing. That is the closed loop, and it runs in both
directions (see §2).

### Implementation sketch

1. Fit `a, b, c, const` by OLS on Round 1 with the lag-1 alignment (regress on
   differences to avoid a spurious levels fit; check with non-negativity constraints
   since these are physical costs). Expect a very high R².
2. Each day: recover `L_{t-1}` from the identity, Kalman-update it with the noisy
   `MenuDash_t` observation to get `L̂_t`, then forecast
   `Ŝizzle_{t+1} = a·Bread_t + b·Sausage_t + c·L̂_t + const`.
3. Position = `±3000` scaled by forecast confidence / predicted move size.

### Capacity

3,000 units, ~$0.15 mean absolute daily move ⇒ ~$450/day ⇒ **~$160k/year** if the
sign is reliably right. Uses 20% of the budget. Highest risk-adjusted return in the
set if the identity holds.

### Tests

- Cross-correlation of `Sizzle` against `Bread` / `Sausage` / `MenuDash` at lags −5…+5;
  the peak should sit cleanly at lag 1.
- R² and residual autocorrelation of the fitted recipe. Residuals should look like a
  smooth latent, not white noise.
- Day-of-week dummies in the regression ("on the tongs", weekend sizzles).
- **Speculative:** the blurb mentions a "sizzle + drink" deal, and UQ Dollar is "the
  primary medium for student transactions for food and drinks". Test whether
  `UQD_{t-1} − 100` enters the sizzle residual. Low probability, cheap to check, and
  if it hits it is *predictable* contamination because UQD reverts to its peg.
- "Offer both tomato and BBQ sauce" is probably flavour text, but could be a two-part
  constant. Low priority.

---

## 2. MenuDash — noisy observation of a smooth latent

> "The fee isn't really about delivery. Most of the couriers are students, and a fair
> few of them spend their weekends behind a folding table in a hardware-store
> carpark; so, when their time becomes more valuable, the fee climbs with it. It has
> been climbing all year. … the app rounds, surges and reshuffles its fees constantly,
> so any given day's number is a **rough read on the thing underneath** rather than an
> exact one. **MenuDash posts a price every day. The cost driving it never does.**"

"a folding table in a hardware-store carpark" is the Bunnings sizzle. The spec is
telling you plainly that the MenuDash courier and the sizzle volunteer are the same
person, i.e. the same latent `L`.

**Model:** `MenuDash_t = α + β·L_t + ε_t`, with `L` smooth/stepwise and `ε` white
observation noise. The last sentence is the giveaway: the *posted* price moves every
day, the *driver* does not.

### Three separable trades

1. **Filter reversion (standalone).** Deviation of `MenuDash_t` from an EWMA/Kalman
   estimate of `L` should revert. 75,000 units × a $0.02 dislocation = **$1,500 per
   reversion**. This is the highest-capacity trade on the board — and also the
   hungriest, at 24.6% of budget.
2. **Sizzle-implied fair value (cross-instrument).** Use `L_{t-1}` backed out of the
   sizzle identity, which has *no observation noise*, as the anchor instead of a
   filter of MenuDash's own noisy history. Strictly more information than (1).
3. **Drift / cycle.** "Climbing all year" suggests a long tilt (~+11% over Round 1,
   ≈ $15k on a full position). But the chart is visibly *cyclical*, not monotone —
   see §9. Trade the seasonal slope of `L`, not a blind long.

### Trap

The noise is what you are harvesting, so an over-smoothed filter kills the edge and an
under-smoothed one trades noise against itself. Fit the filter bandwidth by
maximising realised P&L on Round 1, not by minimising fit error.

---

## 3. Liferaft Ticket — a repeated minority game, and a portfolio hazard

No price history. Price starts at $100,000, limit ±1, floor $20,000, no stated cap.
Majority long ⇒ price **falls $5,000**. Majority short ⇒ price **rises $8,000**. Tie
or all-flat ⇒ unchanged. Flat teams are excluded from the count entirely.

### Payoff structure

Let `p` = probability the majority goes long on a given day.

| Your side | Majority long | Majority short | EV |
|---|---:|---:|---|
| Long | −5,000 | +8,000 | `8000 − 13000p` |
| Short | +5,000 | −8,000 | `13000p − 8000` |

**Long is the better side unless you believe `p > 8/13 ≈ 61.5%`.** The asymmetric
payoff quietly favours the long side, which is the opposite of the naive "everyone
will be long, so I'll be short" reflex.

### The floor is the whole game

At $20,000 the price cannot fall. Majority long ⇒ Δ = 0; majority short ⇒ Δ = +8,000.
So **at the floor, long is a free option**: strictly non-negative payoff. From
$100,000, sixteen consecutive majority-long days reach it. If the crowd skews long
(likely — most teams will hardcode a buy, and the "ship is sinking" framing pushes
people onto the raft), the series walks down to the floor and then locks: everyone
long, price pinned, nobody earns. Any team that arrives at the floor still short is
simply donating.

### The runaway risk

Majority short ⇒ +$8,000/day with no cap. From $100,000, 63 straight short-majority
days puts the price above $600,000, at which point holding a single unit exceeds the
entire budget. **Anything above ~$600,000 makes the instrument untouchable and, if you
don't check, wipes your whole book for the day.** Even at $300,000 it is half your
budget. This must be a hard guard in the sizing code, evaluated before every other
position.

### Strategy

You can read the crowd off your own price history — `self.data` exposes it as the year
runs. `Δ = −5000` ⇒ majority was long; `Δ = +8000` ⇒ majority was short; `Δ = 0` ⇒ tie,
all flat, or pinned at the floor. Most teams will submit a static or slow rule, so the
majority should be strongly persistent. Proposed rule:

- Price at/near the floor (≤ ~$25,000) ⇒ **long** (free or near-free option).
- Otherwise ⇒ infer yesterday's majority from Δ and take the **opposite** side.
- Above a notional threshold ⇒ **flat**, unconditionally, to protect the budget.
- No history yet (day 0) ⇒ **long**, per the EV asymmetry above.

Expected value if you land on the minority side even 60% of the time is $5–8k/day
against 16.7% of budget — an order of magnitude better than anything else here. It is
also the only instrument where your edge depends on other people rather than on a
generator, so treat the estimate as soft.

---

## 4. Boat Party Ticket — deterministic calendar seasonality

> "Demand follows the semester fairly closely, and nobody wants to be on a boat during
> exams. **The same shape turns up at the same point every year**, though the size of
> each peak isn't identical from one year to the next; **the calendar is the reliable
> part, not the height.**"

This is the cleanest signal in the set and the spec all but writes the code for you.
Two peaks and two troughs per year (two semesters), period ≈ 145–150 days, troughs at
the exam periods, and a flat summer plateau in the last ~50 days.

### The critical detail

Amplitude varies year to year; phase does not. So:

- **Do** trade the seasonal *slope*. P&L is `position × Δprice`, so the optimal
  position at day `d` is proportional to `f(d+1) − f(d)` where `f` is the Round-1
  seasonal profile (a centred moving average or a low-order Fourier fit). Long into the
  seasonal rise, short into the seasonal fall.
- **Don't** trade price-versus-seasonal-*level* as mean reversion. A different peak
  height breaks that immediately and turns a good signal into a fader of a real move.

### Refinements

- Rescale the profile online: regress Round-2 prices observed so far against the
  Round-1 profile for a single amplitude + offset, updated daily. Handles "the size
  isn't identical" without touching the phase.
- **Phase check.** Cross-correlate the first ~30 days of Round 2 against the Round-1
  profile to detect a phase shift before committing size. Cheap insurance.

### Capacity

1,000 units × ~$15 peak-to-trough × ~4 half-cycles ⇒ **up to ~$60k/year** on 7.5% of
budget. Excellent Sharpe contributor.

---

## 5. UQ Dollar — hard peg, fast reversion

> "Pegged to the Australian Dollar at $100 … It might wander off during a busy week on
> campus and is usually **back near a hundred a day or two later**. The peg has held all
> year, **in both directions** where the up-spikes correct just as reliably as the
> down-dips."

- **Anchor is a known constant (100)**, not something to estimate. Hard-code it.
- **Reversion half-life is 1–2 days**, so trade today's deviation and expect the payoff
  tomorrow. The spikes look like single-day events, which means today's price alone is
  a sufficient signal — no lookback needed.
- **Explicitly symmetric.** Don't add a directional tilt; short the spikes as
  confidently as you buy the dips.
- "The peg has held all year" reads as a reassurance about the *generator*, so it
  should hold in Round 2 too. Still, add a cheap safety: if a rolling median drifts
  materially off 100 for a sustained stretch, stand down rather than fight a re-peg.
- Use a deadband so you aren't trading noise, and consider sizing proportional to the
  deviation rather than bang-bang (better Sharpe, similar P&L).
- **"Busy week on campus"** is a volatility hint, not a direction hint: deviations
  should cluster with the campus calendar. Worth testing whether |deviation| tracks the
  same semester factor as the Boat Party (§9) — if so, you can pre-position size rather
  than direction.

**Capacity:** ~$1.5 extreme deviation × 650 units ≈ $975 per full round trip; maybe
50–100 exploitable spikes ⇒ **$30–60k/year** on 10.8% of budget. Modest but among the
most reliable — valuable given Sharpe is the tie-break.

---

## 6. Fintech Token — regime switching, and the biggest tail risk

> "It sits still for weeks at a time, then someone posts something in the group chat and
> it jumps to a new price and stays there. **The stable and volatile periods have
> completely different shapes, so make use of more than one strategy to capture the two
> cases.**"

The organisers are explicitly telling you to build a two-regime model.

**Hypothesised generator:** a mean-reverting (OU-like) process around a level, plus a
jump/level-shift process that re-anchors it. The Round-1 chart shows both instant gaps
*and* multi-day slides, so the "volatile" regime is tradable as momentum, not just as a
gap to survive.

- **Quiet regime:** mean-revert to a short rolling median/mean with a band.
- **Volatile regime:** ride the move. Detect via `|r_t| > k·σ_recent`, or better a CUSUM
  changepoint detector on the level (simple, robust, generalises).
- **Re-anchor after a jump.** "Jumps to a new price and stays there" means the new level
  becomes the new anchor immediately.

**The defensive point matters more than the offensive one.** The catastrophic outcome is
sitting long at 800 on a mean-reversion signal while the level relocates to 500 — that's
$350 × 100 = **$35,000 of loss on one move**, larger than a full year of UQ Dollar
scalping. Regime detection earns its keep primarily by getting you *out*, and only
secondarily by getting you on board.

"Nobody has touched the code since" is a nudge that generator parameters are fixed
between rounds — so volatility thresholds fitted on Round 1 should transfer.

**Capacity:** only 100 units (8.8% of budget), but the moves are enormous. Lumpy P&L;
weak for Sharpe, strong for total profit.

---

## 7. Thrifted Jeans — the trap

> "From $40 at an op-shop bin on a Sunday morning then resold at $90 a year later,
> **perhaps this is a sign of a new, steady, upwards trend** for these old thrifted
> jeans?"

**Treat this sentence as adversarial.** It is the only leading, speculative phrasing in
the whole document ("perhaps… ?"), and the chart flatly contradicts it: 40 → 26 → 95 →
38 → 90. That is large-amplitude cyclical/trending behaviour where the start and end
points happen to look like a trend. A Round-2 series starting near $87 could revert hard.

**Better hypothesis:** persistent multi-week trends (a stochastic/autocorrelated drift),
not a constant drift. The day 148→230 leg (30 → 95) and the day 230→265 leg (95 → 38)
are both strong and sustained, which is a trend-follower's profile, not a buy-and-holder's.

- Momentum with a 10–30 day lookback (MA crossover or breakout), volatility-scaled
  sizing, and a stop.
- Test for a longer cycle too (local extrema look ~65–70 days apart), but unlike the
  Boat Party there is **no calendar anchor in the story**, so don't lock a fixed phase.
- **"Sunday morning"** is the one weekday hint here — worth a day-of-week test on returns.

**Capacity:** 800 units × $50 swings = **~$40k per major swing**, the largest raw upside
of the price-series instruments, at 11.6% of budget. Also the highest variance. Size it
by volatility, not by conviction.

---

## 8. Bread & Sausage — inputs first, positions second

Both are described as slow climbers with noisy short-run behaviour, and both are "on the
same shopping list".

**Bread:** > "prices generally rise over time as flour becomes more expensive. Within a
year, prices tend to increase gradually, but **over the course of a week they often move
in larger, less predictable jumps**."

Three readings of that last clause, all testable and not mutually exclusive:
1. A **day-of-week effect** — the big move lands on a particular weekday (the shopping day).
2. **Variance scaling super-linearly** with horizon ⇒ positive autocorrelation ⇒ weekly momentum.
3. The annual drift is signal, the weekly wiggle is noise ⇒ **mean-revert deviations from a
   slow trend** (30–60 day MA / EWMA). This is the most directly tradable formulation.

**Sausage:** > "the butcher on Hawken Drive, who has had the **same weekend order for
about eleven years**. Same shape as the bread."

The constant order is a hint that quantity is fixed, so the series is a *pure cost*
process with no demand component. "Weekend order" implies a weekly restock cadence — test
for a 7-day cycle in level or volatility.

### The capital-efficiency point

| | Buy-and-hold P&L (R1) | Notional | Return on notional |
|---|---:|---:|---:|
| Sausage | ~$7,000 | 30,750 | **~23%** |
| Bread | ~$3,500 | 63,500 | ~5.5% |

**Sausage is roughly four times more capital-efficient than bread** for the same
"climbs slowly" story, because its position limit is generous relative to its price.
When the budget binds — and it does, by ~16% — bread is the position to cut. Keep bread
as a *data feed* for the sizzle model; it costs nothing to read.

---

## 9. How they connect

### 9a. The cost chain (the designed edge)

```
  Bread_{t-1} ─┐
Sausage_{t-1} ─┼──► Sausage Sizzle_t          (exact, lagged, noiseless)
        L_{t-1}┘
        
        L_t ──────► MenuDash_t + noise         (contemporaneous, noisy)
```

Two directions of information flow, and they close a loop:

- **Forward:** Bread/Sausage/MenuDash observed today ⇒ tomorrow's Sizzle, near-exactly.
- **Backward:** Sizzle observed today ⇒ `L_{t-1}` recovered *exactly* ⇒ a clean fair
  value for MenuDash ⇒ trade MenuDash's observation noise against it.
- Because `L` is smooth, the one-day lag in the backward direction barely costs anything.

The right architecture is a small state-space model: one latent `L`, two measurements
(the noiseless lagged one from the Sizzle, the noisy contemporaneous one from MenuDash),
Kalman-updated daily. Everything else falls out of it.

### 9b. The campus-calendar factor

The Boat Party is explicitly semester-driven. MenuDash's smooth component looks
cyclical on the same rough period (~145 days), with its troughs close to the Boat
Party's and its peaks lagging by a couple of weeks — consistent with a shared "campus
activity" driver: students are busy in semester, their time is worth more, the delivery
fee climbs, and the boat sells out. UQ Dollar's "busy week on campus" deviations plug
into the same calendar, and the Sizzle inherits it through `L`.

**Practical use:** if a single seasonal factor drives the Boat Party level, MenuDash's
latent, and UQ Dollar's *volatility*, you can (a) forecast `L` seasonally rather than
just filtering it, and (b) anticipate UQ Dollar's active periods. Worth testing, and
worth being careful about — it also means these positions are correlated, which matters
for the Sharpe tie-break.

### 9c. Independent

Thrifted Jeans, Fintech Token and the Liferaft share no driver with anything else. The
Liferaft in particular is adversarial rather than statistical — it belongs in a separate
mental bucket entirely.

---

## 10. The weekday/weekend question

Six of nine instruments carry weekly language:

| Instrument | Phrase |
|---|---|
| Sausage | "same **weekend** order for about eleven years" |
| Bread | "over the course of **a week** they often move in larger jumps" |
| MenuDash | "spend their **weekends** behind a folding table" |
| Sausage Sizzle | "on the tongs for **four hours**"; Bunnings sizzles are weekend events |
| Thrifted Jeans | "op-shop bin on a **Sunday morning**" |
| UQ Dollar | "wander off during a **busy week** on campus" |

That is too consistent to be accidental. Concrete tests:

- Mean and σ of returns bucketed by `day % 7`, per instrument.
- Autocorrelation at lag 7; spectral power at frequency 1/7.
- Day-of-week dummies added to the Sizzle recipe regression.

**The catch — and it is a real one.** The mapping from `day` index to weekday is
unknown, and `self.day` resets to 0 in Round 2. If Round 2's day 0 falls on a different
weekday than Round 1's, a phase-locked weekday rule fitted on Round 1 will be wrong by a
fixed offset all year, and a 7-day signal applied at the wrong phase is worse than no
signal at all.

Mitigations, in order of preference:
1. Prefer **phase-agnostic** formulations (e.g. "the day after a large bread move",
   rather than "Tuesdays").
2. **Re-estimate the phase online** in Round 2 from the first few weeks before scaling up.
3. Only phase-lock if the Round-1 effect is very strong *and* you accept the risk.

---

## 11. Ranked test plan

| # | Test | Why it's first |
|---|---|---|
| 1 | Lag-1 regression of Sizzle on Bread/Sausage/MenuDash; check R² and cross-correlogram | Confirms or kills the largest, most reliable edge |
| 2 | Confirm P&L timing convention in `simulation.py` | Everything in #1 depends on it |
| 3 | Back out `L` from the Sizzle residual; check MenuDash reverts to it | Unlocks the highest-capacity trade |
| 4 | UQ Dollar reversion to 100: deviation half-life, deadband sweep | Simplest, most certain P&L; good Sharpe |
| 5 | Boat Party seasonal profile + slope trade; phase cross-correlation | Spec explicitly promises it transfers |
| 6 | Day-of-week sweep across all nine | Cheap; six instruments hint at it |
| 7 | Fintech Token regime detector (CUSUM / vol threshold) | Mostly defensive — caps the worst single-day loss |
| 8 | Thrifted Jeans momentum lookback sweep | Largest raw upside, largest variance |
| 9 | Budget allocator with the Liferaft guard evaluated first | A budget breach zeroes *everything* |

## 12. Traps to keep in view

1. **Thrifted Jeans' "steady upwards trend"** — leading language, contradicted by its own chart.
2. **Bread as a position** — 10.6% of budget for the weakest standalone edge. Read it, don't hold it.
3. **The Liferaft blowing the budget** — a runaway short-majority regime can push the price
   past $600,000, at which point holding one unit wipes your entire book for the day.
4. **Over-fitting the Boat Party amplitude** — the spec warns the height varies. Trade slope, not level.
5. **Day-of-week phase misalignment between rounds** — see §10.
6. **Mean-reverting the Fintech Token through a regime break** — the single largest loss event available.
7. **Correlated positions via the campus factor** — several signals may fire together, which
   concentrates risk exactly when Sharpe is the tie-break.
