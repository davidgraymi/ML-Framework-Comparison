"""Tests for scripts/roll_version.py."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from roll_version import SemVer, analyze_commits, determine_next_version


def test_semver_parse_and_str():
    sv = SemVer.parse("v1.2.3")
    assert sv == SemVer(1, 2, 3)
    assert str(sv) == "1.2.3"
    assert sv.tag == "v1.2.3"

    sv2 = SemVer.parse("0.1.0")
    assert sv2 == SemVer(0, 1, 0)
    assert sv2.tag == "v0.1.0"

    assert SemVer.parse("invalid") is None


def test_semver_bumps():
    sv = SemVer(1, 2, 3)
    assert sv.bump_patch() == SemVer(1, 2, 4)
    assert sv.bump_minor() == SemVer(1, 3, 0)
    assert sv.bump_major() == SemVer(2, 0, 0)


def test_analyze_commits_conventional():
    # Patch cases
    assert analyze_commits("fix: resolve bug") == "patch"
    assert analyze_commits("chore: bump deps\ndocs: fix typo") == "patch"

    # Minor cases
    assert analyze_commits("feat: add fancy capability") == "minor"
    assert analyze_commits("feat(core): new model adapter\nfix: bug") == "minor"

    # Major cases
    assert analyze_commits("feat!: redesign api") == "major"
    assert analyze_commits("fix(api)!: break backward compatibility") == "major"
    assert analyze_commits("chore: update\n\nBREAKING CHANGE: removed deprecated method") == "major"


def test_determine_next_version_in_git_repo(tmp_path: Path):
    def run_git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    # Initialize git repo
    run_git("init")
    run_git("config", "user.name", "Test User")
    run_git("config", "user.email", "test@example.com")

    # Initial commit
    (tmp_path / "file.txt").write_text("hello")
    run_git("add", ".")
    run_git("commit", "-m", "initial commit")

    # 1. No tags yet -> initial version
    rolled, cur, next_v, tag, bump = determine_next_version(
        cwd=str(tmp_path), initial_version="0.1.0"
    )
    assert rolled is True
    assert next_v == "0.1.0"
    assert tag == "v0.1.0"
    assert bump == "initial"

    # Tag the repo with v0.1.0
    run_git("tag", "-a", "v0.1.0", "-m", "Release v0.1.0")

    # 2. HEAD is at v0.1.0 -> no roll
    rolled, cur, next_v, tag, bump = determine_next_version(cwd=str(tmp_path))
    assert rolled is False
    assert cur == "0.1.0"
    assert bump == "none"

    # 3. Add fix commit -> patch bump to 0.1.1
    (tmp_path / "file.txt").write_text("hello world")
    run_git("add", ".")
    run_git("commit", "-m", "fix: small bugfix")

    rolled, cur, next_v, tag, bump = determine_next_version(cwd=str(tmp_path))
    assert rolled is True
    assert cur == "0.1.0"
    assert next_v == "0.1.1"
    assert tag == "v0.1.1"
    assert bump == "patch"

    # Tag v0.1.1
    run_git("tag", "-a", "v0.1.1", "-m", "Release v0.1.1")

    # 4. Add feat commit -> minor bump to 0.2.0
    (tmp_path / "file.txt").write_text("hello universe")
    run_git("add", ".")
    run_git("commit", "-m", "feat: new capability")

    rolled, cur, next_v, tag, bump = determine_next_version(cwd=str(tmp_path))
    assert rolled is True
    assert cur == "0.1.1"
    assert next_v == "0.2.0"
    assert tag == "v0.2.0"
    assert bump == "minor"

    # Tag v0.2.0
    run_git("tag", "-a", "v0.2.0", "-m", "Release v0.2.0")

    # 5. Add breaking commit -> major bump to 1.0.0
    (tmp_path / "file.txt").write_text("bye")
    run_git("add", ".")
    run_git("commit", "-m", "feat!: breaking change")

    rolled, cur, next_v, tag, bump = determine_next_version(cwd=str(tmp_path))
    assert rolled is True
    assert cur == "0.2.0"
    assert next_v == "1.0.0"
    assert tag == "v1.0.0"
    assert bump == "major"
