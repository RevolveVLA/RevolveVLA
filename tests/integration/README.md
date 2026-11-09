# tests/integration
The manager loop against stub components: `./revolve test manager` (with the manager's unit tests) or
`./revolve test integration`. stub_steps.py replaces each step's work (REVOLVE_STUB=<script.json>); the manager
and step containers, the wrapper's HTTP, labels, stall kill, retry and kill + resume are real. Runs under
runs/mt_*/ and runs/manager_tests/ (git-ignored). Needs the revolve-manager image (`./revolve build manager`).
