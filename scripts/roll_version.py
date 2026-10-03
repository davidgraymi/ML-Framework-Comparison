#!/usr/bin/env python3
"""Automatic version rolling utility for neural-cost.

Determines the next semantic version based on git history and Conventional Commits.
Can create git tags and report outputs for GitHub Actions.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

SEMVER_REGEX = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
CONVENTIONAL_BREAKING = re.compile(r"^(\w+)(\([^)]+\))?!:|BREAKING CHANGE:", re.MULTILINE)
CONVENTIONAL_FEAT = re.compile(r"^feat(\([^)]+\))?:", re.IGNORECASE)


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> SemVer | None:
        match = SEMVER_REGEX.match(version_str.strip())
        if not match:
            return None
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def bump_major(self) -> SemVer:
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> SemVer:
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemVer:
        return SemVer(self.major, self.minor, self.patch + 1)

    @property
    def tag(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def run_git(args: list[str], cwd: str | None = None) -> str:
    """Run a git command and return stripped stdout."""
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def get_all_version_tags(cwd: str | None = None) -> list[tuple[SemVer, str]]:
    """Return all valid semver tags sorted in ascending order."""
    try:
        raw_tags = run_git(["tag", "-l"], cwd=cwd).splitlines()
    except subprocess.CalledProcessError:
        return []

    parsed: list[tuple[SemVer, str]] = []
    for tag in raw_tags:
        sv = SemVer.parse(tag)
        if sv is not None:
            parsed.append((sv, tag))

    parsed.sort(key=lambda item: (item[0].major, item[0].minor, item[0].patch))
    return parsed


def get_head_sha(cwd: str | None = None) -> str:
    return run_git(["rev-parse", "HEAD"], cwd=cwd)


def get_commit_sha(rev: str, cwd: str | None = None) -> str:
    return run_git(["rev-parse", f"{rev}^{{commit}}"], cwd=cwd)


def analyze_commits(commits_log: str) -> str:
    """Analyze commit log text to determine bump type: 'major', 'minor', or 'patch'."""
    if CONVENTIONAL_BREAKING.search(commits_log):
        return "major"
    for line in commits_log.splitlines():
        if CONVENTIONAL_FEAT.search(line):
            return "minor"
    return "patch"


def determine_next_version(
    cwd: str | None = None,
    initial_version: str = "0.1.0",
) -> tuple[bool, str, str, str, str]:
    """Determine whether version should roll and what the new version is.

    Returns:
        (rolled, current_version, next_version, next_tag, bump_type)
    """
    version_tags = get_all_version_tags(cwd=cwd)
    head_sha = get_head_sha(cwd=cwd)

    if not version_tags:
        # No tags exist yet; initialize base version
        sv = SemVer.parse(initial_version) or SemVer(0, 1, 0)
        return True, "0.0.0", str(sv), sv.tag, "initial"

    latest_semver, latest_tag = version_tags[-1]
    latest_tag_sha = get_commit_sha(latest_tag, cwd=cwd)

    if head_sha == latest_tag_sha:
        # HEAD is already tagged at the latest version
        return False, str(latest_semver), str(latest_semver), latest_tag, "none"

    # There are changes between latest_tag and HEAD
    commit_range = f"{latest_tag}..HEAD"
    commits_log = run_git(["log", commit_range, "--format=%s%n%b"], cwd=cwd)

    bump = analyze_commits(commits_log)
    if bump == "major":
        next_sv = latest_semver.bump_major()
    elif bump == "minor":
        next_sv = latest_semver.bump_minor()
    else:
        next_sv = latest_semver.bump_patch()

    return True, str(latest_semver), str(next_sv), next_sv.tag, bump


def write_github_output(outputs: dict[str, str]) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        f.writelines(f"{k}={v}\n" for k, v in outputs.items())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll package version based on git changes.")
    parser.add_argument("--initial", default="0.1.0", help="Initial version if no tags exist")
    parser.add_argument("--dry-run", action="store_true", help="Calculate version without tagging")
    parser.add_argument("--tag", action="store_true", help="Create the git tag if version rolled")
    parser.add_argument(
        "--push", action="store_true", help="Push the tag to remote (used with --tag)"
    )
    parser.add_argument("--remote", default="origin", help="Remote to push to (default: origin)")
    args = parser.parse_args(argv)

    rolled, cur_ver, next_ver, next_tag, bump = determine_next_version(initial_version=args.initial)

    outputs = {
        "rolled": "true" if rolled else "false",
        "current_version": cur_ver,
        "next_version": next_ver,
        "next_tag": next_tag,
        "bump_type": bump,
    }

    write_github_output(outputs)

    if not rolled:
        print(f"No changes since latest tag {next_tag}. Version remains {cur_ver}.")
        return 0

    print(f"Version roll detected [{bump}]: {cur_ver} -> {next_ver} ({next_tag})")

    if args.tag and not args.dry_run:
        print(f"Creating git tag {next_tag}...")
        subprocess.run(
            ["git", "tag", "-a", next_tag, "-m", f"Release {next_tag}"],
            check=True,
        )
        if args.push:
            print(f"Pushing {next_tag} to {args.remote}...")
            subprocess.run(
                ["git", "push", args.remote, next_tag],
                check=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
