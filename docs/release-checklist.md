# Release checklist

## 0.9.0

- Review the complete local diff and commit it on a release branch.
- Open a pull request and require the Julia 1.10/1.12 Linux, macOS, Windows,
  Aqua/JET, lower-bound, coverage, and documentation jobs to pass.
- Confirm a fresh environment can load FluxKAN without Lux or MLDatasets.
- Review the generated documentation and benchmark baseline.
- Merge the release pull request, then close obsolete PRs #11 and #15 as
  superseded and close the completed 0.1 registration issue #13.
- Run Registrator for version 0.9.0 and confirm TagBot creates `v0.9.0`.
- Verify the tagged documentation appears under `stable` and that a registry
  install reports `FluxKAN v0.9.0`.

Do not create the tag before the multi-platform CI has run on the pushed
commit. Local verification covers Julia 1.12.6/Linux; the remaining platforms
are deliberately delegated to the release pull request's CI matrix.

## Toward 1.0

Use 0.9 for real training workloads and collect feedback specifically about
constructor names, feature-first layout, boundary behavior, grid-update
semantics, and serialization expectations. Make any required breaking changes
in another 0.x release; tag 1.0 only when those contracts no longer need to
change.
