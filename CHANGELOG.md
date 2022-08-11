CHANGELOG
=========

2022-06-13 (v3.1.0)
-------------------
- Added TI parameter set.

2022-03-01 (v3.0.2)
-------------------
- changes to `docs/Makefile` to move contents to common `Makefile`
- adapted CI to allow for new proxy server
- added deploy command for documentation to `setup.py`

2021-25-10 (v3.0.1)
-------------------
- Bugfix in implementation of faulty measurements. Now using NetSquid's IMeasureFaulty.

2021-07-09 (v3.0.0)
-------------------
- `IonTrap` now only takes positional arguments (no `**kwargs`).
- `IonTrap` no longer takes the argument `"noiseless"`; instead, all parameters always default to their noiseless value.
- `IonTrap` now takes arguments `"prob_error_0"` and `"prob_error_1"` instead of `"prop_error_0"` and `"prop_error_1"` (this was a typo).
- `IonTrap` documentation has been improved.
- Removed coupling to phonons from modeling of MS gate. Since the model of the gate was only correct for zero temperature anyhow, it is better not to model noise this way and instead use error models (the phonon coupling was hardcoded into the instruction itself).

2021-03-12 (v2.0.0)
-------------------
 - Upgraded to netsquid 1.0.
 - Removed protocols, subprotocols, sequential task executor and simulation code.

2021-03-12 (v1.1.0)
-------------------
 - Added protocols and simulation scripts for two-node and three-node simulations. Also readded old version of DoubleClickMagicDistributor, and added old version of QDetector, to avoid dependency problems in simulations.


2020-11-11 (v1.0.1)
-------------------
 - Removed dependency on pyparsing.
 - Fix in version tracking.

2020-07-15 (v1.0.0)
-------------------
 - Removed double-click magic (moved to netsquid-magic).
 - MagicEntGenSubProt now takes a magic distributor as argument, rather than a magic-distributor adaptor.
