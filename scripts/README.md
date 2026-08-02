# Windows workflow launchers

The launchers in `syncwb/` record concrete operational SyncWB commands. They
are examples and reproducible local recipes, not an alternative command-line
interface.

Every launcher changes to the repository root before invoking `syncwb`.
Consequently, paths such as `workbench.sqlite`, `artifact_store`, and
`reports` resolve consistently even when a launcher is called from another
working directory.

The files are intentionally separate where they invoke different SyncWB
commands. Subject- or experiment-specific launchers include that scope in their
filename.

PyTorch-dependent launchers live separately in `../ml/experiments/`.
