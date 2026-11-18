#!/usr/bin/env bash

# CHANGELOG.md, crates/*/CHANGELOG.md, MIGRATING.md and docs/migrations/ are
# release documents: they are written on the release PR, from the merged pull
# requests' `## Changelog` / `## Migration` sections and the public API diff.
# An ordinary PR never edits them — that is what makes several in-flight PRs
# rebase cleanly. This guard fails a PR whose diff against its base touches
# any of them. Release PRs (`release-plz-*` head branch or the `release`
# label) are exempted by the workflow's job condition, not here.
#
# A fork does not write these documents: it takes them from this repository
# when it merges a release. There the base is the fork's own line, and a PR
# that merges a release differs from it in every release document the
# release wrote. So where the documents come from is named separately from
# the base (UPSTREAM_REF, by default the base). A release document may differ
# from the base only by being, byte for byte, what the newest commit of that
# history contained in HEAD says it is: a merged release passes, and any
# other edit of a release document fails as it does here. In this repository
# the two refs are the same and the rule is the one above.

set -euo pipefail

base_ref="${1:-${BASE_REF:-origin/main}}"
readonly base_ref
readonly frozen_pattern='^(CHANGELOG\.md|MIGRATING\.md|crates/[^/]+/CHANGELOG\.md|docs/migrations/.+)$'
readonly canonical_repository='0xPlaygrounds/rig'

if ! merge_base="$(git merge-base "${base_ref}" HEAD)"; then
    echo "Cannot find the merge base of ${base_ref} and HEAD." >&2
    exit 1
fi

# In a fork's CI the documents' history is this repository's main branch.
upstream_ref="${UPSTREAM_REF:-}"
if [[ -z "${upstream_ref}" && -n "${GITHUB_REPOSITORY:-}" && "${GITHUB_REPOSITORY}" != "${canonical_repository}" ]]; then
    git fetch --no-tags --quiet "https://github.com/${canonical_repository}.git" main
    upstream_ref="$(git rev-parse FETCH_HEAD)"
fi

upstream_base=""
if [[ -n "${upstream_ref}" ]]; then
    if ! upstream_base="$(git merge-base "${upstream_ref}" HEAD)"; then
        echo "Cannot find the merge base of ${upstream_ref} and HEAD." >&2
        exit 1
    fi
fi

# Every path that differs, both sides of a rename: with rename detection a
# release document deleted here would be listed only under the name of an
# identical file the release added. Names are read NUL-separated and matched
# literally, so a name is what it is whatever characters it holds. The
# listing is written out first so that a failure to produce it fails here.
changed="$(mktemp)"
trap 'rm -f "${changed}"' EXIT
git diff --no-renames --name-only -z "${merge_base}" HEAD > "${changed}"

touched=()
while IFS= read -r -d '' file; do
    [[ "${file}" =~ ${frozen_pattern} ]] || continue
    # The same bytes as the release history has, or absent from both.
    if [[ -n "${upstream_base}" ]] \
        && git --literal-pathspecs diff --quiet "${upstream_base}" HEAD -- "${file}"; then
        continue
    fi
    touched+=("${file}")
done < "${changed}"

if ((${#touched[@]} > 0)); then
    {
        echo "This PR edits release documents that are generated on the release PR:"
        printf '  %s\n' "${touched[@]}"
        echo
        echo "Do not edit CHANGELOG.md, crates/*/CHANGELOG.md, MIGRATING.md or docs/migrations/"
        echo "in an ordinary PR. Put changelog bullets under '## Changelog' and migration notes"
        echo "under '## Migration' in the PR description instead. The files are regenerated on"
        echo "the release PR by scripts/release-notes.sh and the editorial pass in"
        echo "scripts/release-notes-prompt.md."
    } >&2
    exit 1
fi

if [[ -n "${upstream_base}" ]]; then
    echo "No frozen release docs edited since ${merge_base:0:9}; any that differ are as ${upstream_base:0:9} has them."
else
    echo "No frozen release docs touched since ${merge_base:0:9}."
fi
