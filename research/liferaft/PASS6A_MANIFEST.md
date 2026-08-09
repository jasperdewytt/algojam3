# Liferaft Pass 6A manifest

Status: frozen development-only evidence; no blind or final suite was created or executed.

- Python: `3.12.13 (main, Mar  3 2026, 15:01:35) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`
- CPU count observed: `16`
- Workers: `15`
- Commands: `python -m research.liferaft.pass6_experiments --null --workers auto; python -m research.liferaft.pass6_experiments --null --workers auto; python -m research.liferaft.pass6_experiments --development --workers auto; python -m research.liferaft.pass6_experiments --validation --workers auto; python -m research.liferaft.pass6_experiments --null --workers auto; python -m research.liferaft.pass6_experiments --development --workers auto; python -m research.liferaft.pass6_experiments --validation --workers auto; python -m research.liferaft.pass6_sensitivity --workers auto`
- Runtimes seconds: `{"development": 4.557797399989795, "null_and_power": 195.435883499973, "pivotal_sensitivity": 182.7088986999879, "validation": 148.2572383999941}`
- Serial/parallel bit-identical: `True`

## Result-affecting source hashes

| file | SHA-256 |
|---|---|
| `pass6_models.py` | `2e49b4b901e85c77611dcf577588523015bc001291ee7e002a3e297e6a1437cb` |
| `pass6_strategies.py` | `b0a611a1241aec57eccae6def6426962997414fb7847d139e99c904a8e9807b6` |
| `pass6_experiments.py` | `66228aac256efca171826a43cb925120abf76b8c34c865577a8c39a9c664f489` |
| `pass6_sensitivity.py` | `d80f84f7047786a3f5627f9106573d535098dfb7fbfdb7d3a0136bc2f1bd8489` |
| `test_pass6.py` | `e394677edada57c7298a4714a6ee2a61dccf22ff07e1ef18fa56086b4015814e` |
| `PASS6A_PROTOCOL.md` | `a7816052b786fd2eefa4b207f375d38ba95410b540d7cf74528207e2fe099732` |
| `simulator.py` | `5b61ca8c03e55444a656f5bd5c369bc687494cbed7b13d52b946ed45bbb55dbc` |
| `archetypes.py` | `acda007cf809e8ea76f9792badeedff73acbe4c46a4b59de8522dd10939acecb` |
| `pass3_scenarios.py` | `3ae0cc4daa882078b3ea6ad8db79145c4ecb5642385b746d5ceb5a9635c93983` |
| `cold_start_strategies.py` | `321ae0c06dc168f5615f504ab4b4187d82e222a197bff5880422096a6261ec34` |
| `pass4_strategies.py` | `9d9dd76a3f22569f5d729277f2f23a3ccfc0660720d23f801c61c44e80e0c399` |
| `shadow_strategies.py` | `5593c81f35b7b94c2b8a2177cd9dbdddf054be677e053347b86a1ff3ac61923c` |

## Result artifact hashes

| file | SHA-256 |
|---|---|
| `PASS6A_RESULTS.json` | `db4e64f8e013a7f448e8f180b31fb7e0f4dbf79d4f27fb970fe113d22a76f7d4` |
| `PASS6A_SENSITIVITY.json` | `e04495f9bde5de8dc88b059865e612b0aeb649c38ea6baeae297543d3819783d` |

## Source inputs used

- Existing non-final simulator, archetypes, scenario constructors, cold-start strategies, Pass 4A wrapper, and Pass 5 shadow wrapper.
- Existing `development_scenarios()` and `validation_scenarios()` only; no `final_scenarios()` call.
- No Pass 3/4/5 result was used for tuning; prior reports/results were consumed development evidence.
- `PASS6A_SENSITIVITY.json` contains the separate 0%/5%/10%/20% pivotal-haircut diagnostic; SHA-256: `e04495f9bde5de8dc88b059865e612b0aeb649c38ea6baeae297543d3819783d`.

## Quarantine confirmation

The locked final artifacts remained untouched: no final strategy source, final result, final decision, final execution receipt, final report, final scenario constructor, or `--final` command was opened, parsed, imported, executed, recreated, renamed, overwritten, or used.

Parent process was the sole writer of Pass 6A results, report, and manifest files; worker processes returned data only.
