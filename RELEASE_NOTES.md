# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-04-03

### Added
- `SimulationSeries` can now be initialized as an empty collection (`series_dir=None`) and populated manually via `add_simulation`.
- `add_simulation` and `remove_simulation` now accept either a single string or a list of strings.
- Added a dedicated `remove_simulation` API with a backward-compatible `remove` alias.
- Added regression tests for `SimulationSeries` add/remove behavior and multi-key history extraction.

### Changed
- `Simulation` initialization now expects a single `simulation_dir` path.
- Release packaging now excludes `tests/` from source distributions while keeping tests in the repository.

### Fixed
- Fixed `SimulationSeries.add_history_data` to correctly process multiple history keys without mutating lists during iteration.
- Replaced critical bare `except:` blocks with explicit exception handling in core paths.
- Replaced critical `os.system("rm -r ...")` calls with safe filesystem operations.

## [1.1.1] - 2025-10-16

### Added
### Changed
### Deprecated
### Removed
### Fixed
- Fix interpolation method to handle decreasing x data correctly.
### Security