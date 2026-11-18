"""The release-document guard, in the two repositories it runs in.

`.github/scripts/check-frozen-release-docs.sh` is upstream's guard with one
addition for a fork: a release document may differ from the PR's base when it
is byte for byte what the merged upstream history says it is. These tests
build small repositories and run the script on them. No network: the upstream
ref is always given.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check-frozen-release-docs.sh"


class Repo:
    def __init__(self, root):
        self.root = Path(root)
        self.git("init", "--quiet", "--initial-branch=main")

    def git(self, *args):
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
        }
        done = subprocess.run(
            ["git", *args], cwd=self.root, env=env, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return done.stdout.strip()

    def commit(self, message, files=None, remove=()):
        for path, text in (files or {}).items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        for path in remove:
            self.git("rm", "--quiet", path)
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def guard(self, base, upstream=None, repository=None):
        env = {key: value for key, value in os.environ.items()
               if key not in ("UPSTREAM_REF", "BASE_REF", "GITHUB_REPOSITORY")}
        env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
        if upstream is not None:
            env["UPSTREAM_REF"] = upstream
        if repository is not None:
            env["GITHUB_REPOSITORY"] = repository
        return subprocess.run(
            ["bash", str(SCRIPT), base], cwd=self.root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )


def named(done):
    """The files the guard's failure names."""
    return [line.strip() for line in done.stderr.splitlines() if line.startswith("  ")]


RELEASE_ONE = {
    "CHANGELOG.md": "# 1\n",
    "MIGRATING.md": "to 1\n",
    "crates/a/CHANGELOG.md": "a 1\n",
    "docs/migrations/old.md": "old\n",
    "src/lib.rs": "one\n",
}
RELEASE_TWO = {
    "CHANGELOG.md": "# 2\n# 1\n",
    "crates/a/CHANGELOG.md": "a 2\na 1\n",
    "docs/migrations/two.md": "two\n",
    "src/lib.rs": "two\n",
}


class ForkHistory(unittest.TestCase):
    """A fork on release one, and upstream at release two and beyond."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Repo(self.tmp.name)
        repo = self.repo
        self.one = repo.commit("release one", RELEASE_ONE)
        repo.git("branch", "fork")
        self.two = repo.commit(
            "release two", RELEASE_TWO, remove=["docs/migrations/old.md"]
        )
        self.later = repo.commit("after release two", {"CHANGELOG.md": "# unreleased\n# 2\n# 1\n"})
        repo.git("checkout", "--quiet", "fork")
        self.fork_base = repo.commit("the fork's own code", {"addons/x.rs": "x\n"})
        repo.git("checkout", "--quiet", "-b", "pr")

    def merge_release_two(self):
        self.repo.git("merge", "--quiet", "--no-ff", "-m", "merge release two", self.two)

    def test_merging_a_release_carries_its_documents(self):
        self.merge_release_two()
        done = self.repo.guard(self.fork_base, upstream=self.later)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f"as {self.two[:9]} has them", done.stdout)

    def test_the_same_merge_fails_without_the_release_history(self):
        self.merge_release_two()
        done = self.repo.guard(self.fork_base)
        self.assertEqual(done.returncode, 1)
        self.assertEqual(
            named(done),
            ["CHANGELOG.md", "crates/a/CHANGELOG.md",
             "docs/migrations/old.md", "docs/migrations/two.md"],
        )

    def test_an_edit_on_top_of_a_merged_release_fails_and_is_the_only_file_named(self):
        self.merge_release_two()
        self.repo.commit("a note of the fork's", {"crates/a/CHANGELOG.md": "a 2\na 1\nfork\n"})
        done = self.repo.guard(self.fork_base, upstream=self.later)
        self.assertEqual(done.returncode, 1)
        self.assertEqual(named(done), ["crates/a/CHANGELOG.md"])

    def test_an_ordinary_fork_pr_may_not_edit_add_or_delete_a_release_document(self):
        edits = {
            "edit": dict(files={"MIGRATING.md": "edited\n"}),
            "add": dict(files={"docs/migrations/mine.md": "mine\n"}),
            "delete": dict(remove=["docs/migrations/old.md"]),
        }
        expected = {
            "edit": "MIGRATING.md", "add": "docs/migrations/mine.md",
            "delete": "docs/migrations/old.md",
        }
        for name, change in edits.items():
            with self.subTest(name):
                self.repo.git("checkout", "--quiet", "-B", "pr", self.fork_base)
                self.repo.commit(name, **change)
                done = self.repo.guard(self.fork_base, upstream=self.later)
                self.assertEqual(done.returncode, 1)
                self.assertEqual(named(done), [expected[name]])

    def test_copying_a_document_from_unmerged_upstream_fails(self):
        self.repo.commit("copied ahead", {"CHANGELOG.md": "# unreleased\n# 2\n# 1\n"})
        done = self.repo.guard(self.fork_base, upstream=self.later)
        self.assertEqual(done.returncode, 1)
        self.assertEqual(named(done), ["CHANGELOG.md"])

    def test_a_pr_that_touches_no_release_document_passes(self):
        self.repo.commit("code", {"addons/y.rs": "y\n"})
        for upstream in (None, self.later):
            with self.subTest(upstream=upstream):
                done = self.repo.guard(self.fork_base, upstream=upstream)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_in_upstreams_own_repository_any_touch_fails_as_before(self):
        self.repo.git("checkout", "--quiet", "-B", "pr", self.later)
        self.repo.commit("an ordinary PR", {"CHANGELOG.md": "# mine\n"})
        done = self.repo.guard(self.later, repository="0xPlaygrounds/rig")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(named(done), ["CHANGELOG.md"])


if __name__ == "__main__":
    unittest.main()
