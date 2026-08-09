# Liferaft Pass 5A Frozen Development Manifest

- selected challenger: **None**
- exact frozen parameters: `none; no shadow candidate passed the screening rule`
- all frozen shadow candidate parameters:
  - `shadow8_markov`: `{"alpha": 1.0, "candidate_name": "shadow8_markov", "cooldown_genuine_observations": 5, "deactivation_virtual_pnl_limit": -10000, "health_bad_streak_required": 2, "markov_order": 2, "maximum_actual_loss": 50000, "minimum_confidence": 0.1, "minimum_expected_pnl": 1000.0, "minimum_genuine_nonzero_observations": 8, "minimum_initial_virtual_pnl": 10000, "minimum_recent_virtual_pnl": 5000, "minimum_scoreable_virtual_trades": 6, "portfolio_reserve": 10000, "shadow_health_window": 12}`
  - `shadow12_markov`: `{"alpha": 1.0, "candidate_name": "shadow12_markov", "cooldown_genuine_observations": 5, "deactivation_virtual_pnl_limit": -10000, "health_bad_streak_required": 2, "markov_order": 2, "maximum_actual_loss": 50000, "minimum_confidence": 0.1, "minimum_expected_pnl": 1000.0, "minimum_genuine_nonzero_observations": 12, "minimum_initial_virtual_pnl": 10000, "minimum_recent_virtual_pnl": 5000, "minimum_scoreable_virtual_trades": 6, "portfolio_reserve": 10000, "shadow_health_window": 12}`
  - `shadow20_markov`: `{"alpha": 1.0, "candidate_name": "shadow20_markov", "cooldown_genuine_observations": 5, "deactivation_virtual_pnl_limit": -10000, "health_bad_streak_required": 2, "markov_order": 2, "maximum_actual_loss": 50000, "minimum_confidence": 0.1, "minimum_expected_pnl": 1000.0, "minimum_genuine_nonzero_observations": 20, "minimum_initial_virtual_pnl": 10000, "minimum_recent_virtual_pnl": 5000, "minimum_scoreable_virtual_trades": 6, "portfolio_reserve": 10000, "shadow_health_window": 12}`
- experiment command: `python -m research.liferaft.pass5_experiments`
- development scenarios: 9
- consumed validation scenarios: 480
- total scenarios: 489
- candidates per exposure: 7
- exposure levels: 4
- result cells: 13692
- runtime seconds: 1258.26
- screening rule passed: False
- evidence status: development evidence only; not a production acceptance result
- fresh blind Pass 5B suite: not created or executed
- tuning prohibition: no tuning from any previously consumed final result

## Result-affecting source hashes

| source | SHA-256 |
| --- | --- |
| `PASS5A_PROTOCOL.md` | `b9467263fbba476b04309a2074d9b484859f08497efe4e0b908a0a9e959b0dd2` |
| `archetypes.py` | `acda007cf809e8ea76f9792badeedff73acbe4c46a4b59de8522dd10939acecb` |
| `simulator.py` | `5b61ca8c03e55444a656f5bd5c369bc687494cbed7b13d52b946ed45bbb55dbc` |
| `strategies.py` | `9bb4c0d5718d126b4725c2ac52c5afbe774c4e5871f736cb6acdb3c2018b7111` |
| `cold_start_strategies.py` | `321ae0c06dc168f5615f504ab4b4187d82e222a197bff5880422096a6261ec34` |
| `pass3_scenarios.py` | `3ae0cc4daa882078b3ea6ad8db79145c4ecb5642385b746d5ceb5a9635c93983` |
| `pass4_strategies.py` | `9d9dd76a3f22569f5d729277f2f23a3ccfc0660720d23f801c61c44e80e0c399` |
| `shadow_strategies.py` | `5593c81f35b7b94c2b8a2177cd9dbdddf054be677e053347b86a1ff3ac61923c` |
| `pass5_experiments.py` | `707408088405bc601a90e809be51e9c3d6f712a45f4b77d5d0edc43ae2d6c691` |

The protocol was frozen before the full experiment.  Consumed final artifacts
remained quarantined, no final scenario constructor was called, and no
production file or final catalogue was modified.
