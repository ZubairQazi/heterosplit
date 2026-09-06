# Releasing

HeteroSplit publishes to PyPI via GitHub Actions using **trusted publishing** (OIDC — no
API tokens stored in the repo).

## One-time PyPI setup

1. Create the project on PyPI (and, for a dry run, TestPyPI).
2. Add a **trusted publisher** for it (PyPI → project → Publishing):
   - Owner: `ZubairQazi`
   - Repository: `heterosplit`
   - Workflow: `release.yml`
   - Environment: `pypi` (and `testpypi` for the dry run)
3. In the GitHub repo, create matching environments named `pypi` and `testpypi`
   (Settings → Environments).

See <https://docs.pypi.org/trusted-publishers/>.

## Cutting a release

1. Bump `__version__` in `src/heterosplit/__init__.py` (single source of truth; hatchling
   reads it).
2. Move the `Unreleased` notes in `CHANGELOG.md` under a new version heading with today's
   date.
3. Commit, then tag and push:

   ```bash
   git commit -am "release: v0.1.1"
   git tag v0.1.1
   git push && git push --tags
   ```

   The `release.yml` workflow builds the sdist + wheel, verifies the tag matches
   `__version__`, and publishes to PyPI.

## Dry run (TestPyPI)

Use the workflow's **Run workflow** button (Actions → Release) with target `testpypi`
before a real release.
