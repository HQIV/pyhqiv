#!/usr/bin/env python3
"""
hqiv-arena — CLI for the HQIV Arena (GitHub-native physics improvement benchmark).

Modeled after the ecdsafail / Yukon solver CLI pattern.

Commands:
  login [token] [--api ...]
  config
  benchmark
  clone [dir]
  setup
  run
  submit --note-file FILE --model "..." [--claimed-score X]
  submissions [--all]
  note <ref>
  sync
  reset <ref>
  version
  update
  install-skill [--target agents|claude|all]

The CLI detects when it is running inside a cloned HQIV arena workspace
(by looking for marker files / local git config written by `clone`).

It prefers `gh` (GitHub CLI) when available for PR / push operations,
falling back to direct GitHub API calls using the stored PAT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# Auto-insert src/ when running from an unpacked source tree (before any relative imports)
def _auto_insert_src() -> None:
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "src" / "pyhqiv" / "__init__.py").exists():
            src = str(cand / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            break
        if (cand / "pyproject.toml").exists() and (cand / "src" / "pyhqiv").exists():
            src = str(cand / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            break

_auto_insert_src()

# --- Config -----------------------------------------------------------------

APP_NAME = "hqiv-arena"
CONFIG_DIR_NAME = "hqiv-arena"
CONFIG_FILE_NAME = "config.json"
SKILL_NAME = "hqiv-arena"

# Default GitHub OAuth App client_id for device flow (public client).
# In production you would register https://github.com/settings/applications/new
# and use the client_id here. For now we guide users to PATs (simpler + no app approval).
GITHUB_DEVICE_CLIENT_ID = "Iv1.b507a08c7e8e4a2c"  # placeholder; real one would be set

DEFAULT_API_BASE = "https://api.github.com"

CONFIG_ENV_TOKEN = "HQIV_ARENA_TOKEN"
CONFIG_ENV_API = "HQIV_ARENA_API_URL"

MARKER_FILE = ".hqiv-arena"
LOCAL_GIT_CONFIG_KEY = "hqiv-arena.workspace"

# The two repos that make up the HQIV benchmark workspace
HQIV_LEAN_REPO = "https://github.com/HQIV/hqiv-lean.git"
PYHQIV_REPO = "https://github.com/disregardfiat/pyhqiv.git"


@dataclass
class Config:
    token: Optional[str] = None
    api_base_url: str = DEFAULT_API_BASE

    def to_dict(self) -> Dict[str, Any]:
        return {"token": self.token, "apiBaseUrl": self.api_base_url}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        return cls(
            token=d.get("token"),
            api_base_url=d.get("apiBaseUrl", DEFAULT_API_BASE),
        )


def config_dir() -> Path:
    return Path.home() / ".config" / CONFIG_DIR_NAME


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def read_config() -> Config:
    p = config_path()
    if p.exists():
        try:
            return Config.from_dict(json.loads(p.read_text()))
        except Exception:
            pass
    return Config()


def write_config(cfg: Config) -> None:
    config_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    p = config_path()
    p.write_text(json.dumps(cfg.to_dict(), indent=2))
    p.chmod(0o600)


def get_effective_token(cfg: Config) -> Optional[str]:
    return os.environ.get(CONFIG_ENV_TOKEN) or cfg.token


def get_effective_api_base(cfg: Config) -> str:
    return os.environ.get(CONFIG_ENV_API) or cfg.api_base_url


def current_skill_install_targets() -> Dict[str, Path]:
    home = Path.home()
    return {
        "agents": home / ".agents" / "skills" / SKILL_NAME / "SKILL.md",
        "claude": home / ".claude" / "skills" / SKILL_NAME / "SKILL.md",
        "opencode": home / ".config" / "opencode" / "skills" / SKILL_NAME / "SKILL.md",
        "cursor": home / ".cursor" / "skills" / SKILL_NAME / "SKILL.md",
        "grok": home / ".grok" / "skills" / SKILL_NAME / "SKILL.md",
    }


def load_skill_content() -> str:
    """Load the SKILL.md content from package data or a sibling file (dev)."""
    try:
        import importlib.resources as resources

        # After install: pyhqiv/arena/SKILL.md
        pkg_files = resources.files("pyhqiv.arena")
        skill_file = pkg_files / "SKILL.md"
        if skill_file.is_file():
            return skill_file.read_text(encoding="utf-8")
    except Exception:
        pass

    # Dev fallback: look next to this file or in repo root arena/
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "SKILL.md",
        here.parents[3] / "arena" / "SKILL.md",  # repo-root/arena/SKILL.md
        here.parents[4] / "src" / "pyhqiv" / "arena" / "SKILL.md",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8")
    # Last resort: embedded minimal version
    return "# HQIV Arena\n\nSee the full guide in the repository.\n"


# --- Git / gh helpers --------------------------------------------------------

def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {"cwd": cwd}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True
    return subprocess.run(cmd, check=check, **kwargs)


def has_gh() -> bool:
    return shutil.which("gh") is not None


def gh_auth_status() -> bool:
    try:
        run(["gh", "auth", "status"], capture=True, check=True)
        return True
    except Exception:
        return False


def git_config_get(key: str, cwd: Optional[Path] = None, local: bool = True) -> Optional[str]:
    try:
        args = ["git", "config"]
        if local:
            args.append("--local")
        args += ["--get", key]
        cp = run(args, cwd=cwd, capture=True, check=False)
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return None


def git_config_set(key: str, value: str, cwd: Optional[Path] = None, local: bool = True) -> None:
    args = ["git", "config"]
    if local:
        args.append("--local")
    args += [key, value]
    run(args, cwd=cwd, check=True)


def is_hqiv_workspace(cwd: Optional[Path] = None) -> bool:
    cwd = cwd or Path.cwd()
    if (cwd / MARKER_FILE).exists():
        return True
    # Also accept if we are inside a tree that has both hqiv-lean and pyhqiv style dirs
    if (cwd / "hqiv-lean" / "lakefile.toml").exists() or (cwd / "pyhqiv" / "pyproject.toml").exists():
        return True
    # Or if git config marker was written
    if git_config_get(LOCAL_GIT_CONFIG_KEY, cwd=cwd):
        return True
    return False


def ensure_clean_worktree(cwd: Optional[Path] = None, force: bool = False) -> None:
    cwd = cwd or Path.cwd()
    try:
        cp = run(["git", "status", "--porcelain"], cwd=cwd, capture=True, check=True)
        if cp.stdout.strip() and not force:
            print("Error: dirty worktree. Commit, stash, or use --force.", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError:
        pass  # not a git repo or other; proceed with caution


# --- Login (GitHub PAT + optional device flow guidance) ----------------------

def do_login(token: Optional[str], api: Optional[str]) -> None:
    cfg = read_config()
    if api:
        cfg.api_base_url = api

    if not token:
        print("HQIV Arena uses a GitHub Personal Access Token (PAT) with 'repo' scope.")
        print("This lets the CLI push branches and open PRs on your behalf.")
        print()
        print("1. Go to: https://github.com/settings/tokens/new")
        print("2. Name it 'HQIV Arena', select the 'repo' scope.")
        print("3. Generate token and paste it below.")
        print()
        print("Alternatively, if you have the GitHub CLI installed and logged in:")
        print("   gh auth login")
        print("   hqiv-arena login   # will try to use your gh token")
        print()
        token = input("Paste GitHub token (ghp_... or github_pat_...): ").strip()

    if not token:
        print("No token provided.", file=sys.stderr)
        sys.exit(1)

    token = token.strip()
    cfg.token = token

    # Verify very lightly (we don't want to require extra scopes for /user if possible)
    try:
        # Use gh if present and authed, else direct
        if has_gh() and gh_auth_status():
            print("Using existing gh authentication where possible.")
        else:
            # Simple validation: hit a public-ish endpoint with the token
            req = urllib.request.Request(
                f"{get_effective_api_base(cfg)}/user",
                headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                user = json.loads(resp.read())
                print(f"Token validated for GitHub user: {user.get('login')}")
    except Exception as e:
        print(f"Warning: could not fully validate token against GitHub ({e}). Storing anyway.")

    write_config(cfg)
    print(f"Logged in. Config saved to {config_path()}")


# --- Clone -------------------------------------------------------------------

def do_clone(target: Optional[str]) -> None:
    target = target or "hqiv-arena-workspace"
    root = Path(target).resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Cloning HQIV Arena workspace into {root} ...")

    lean_dir = root / "hqiv-lean"
    py_dir = root / "pyhqiv"

    if not lean_dir.exists():
        print("Cloning hqiv-lean ...")
        run(["git", "clone", "--depth", "1", HQIV_LEAN_REPO, str(lean_dir)], cwd=root)
    else:
        print("hqiv-lean already present, skipping clone.")

    if not py_dir.exists():
        print("Cloning pyhqiv ...")
        run(["git", "clone", "--depth", "1", PYHQIV_REPO, str(py_dir)], cwd=root)
    else:
        print("pyhqiv already present, skipping clone.")

    # Set up canonical dev symlinks inside pyhqiv (as seen in the real dev layout)
    # pyhqiv expects a sibling or linked hqiv-lean / HQIV_LEAN in some places.
    try:
        (py_dir / "hqiv-lean").symlink_to(lean_dir, target_is_directory=True)
    except FileExistsError:
        pass
    except OSError:
        # On some FS (or Windows) symlink may fail; create a marker instead
        (py_dir / ".hqiv-lean-link").write_text(str(lean_dir))

    # Also help the overlay loader find Hqiv/
    # The real layout often has HQIV_LEAN/hqiv-lean/Hqiv , so make a convenience link
    try:
        if (lean_dir / "hqiv-lean" / "Hqiv").exists():
            (py_dir / "HQIV_LEAN").symlink_to(lean_dir, target_is_directory=True)
    except Exception:
        pass

    # Write workspace marker
    (root / MARKER_FILE).write_text("hqiv-arena workspace\n")
    (py_dir / MARKER_FILE).write_text("hqiv-arena workspace (py side)\n")

    # Write local git config so `hqiv-arena` commands know they are in an arena tree
    try:
        git_config_set(LOCAL_GIT_CONFIG_KEY, "1", cwd=py_dir)
        git_config_set(LOCAL_GIT_CONFIG_KEY, "1", cwd=lean_dir)
    except Exception:
        pass

    print()
    print("Clone complete.")
    print("Next steps:")
    print(f"  cd {py_dir}")
    print("  hqiv-arena setup")
    print("  hqiv-arena run")
    print()
    print("All further hqiv-arena commands should be run from inside the pyhqiv directory (or the workspace root).")


# --- Setup / Run -------------------------------------------------------------

def do_setup(cwd: Optional[Path] = None) -> None:
    cwd = cwd or Path.cwd()
    py_root = cwd if (cwd / "pyproject.toml").exists() else cwd / "pyhqiv"

    print("Running HQIV Arena setup (installing pyhqiv editable + dev extras)...")
    try:
        run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], cwd=py_root)
    except subprocess.CalledProcessError as e:
        print(f"pip install had issues (continuing): {e}")

    # Try to make sure the arena scoring bits are importable
    print("Verifying arena modules...")
    try:
        run([sys.executable, "-c", "from pyhqiv.arena import build_default_metrics, compute_score; print('arena OK')"], cwd=py_root)
    except Exception as e:
        print(f"Warning: arena import check failed: {e}")

    print("Setup done. You can now run `hqiv-arena run`.")


def do_run(cwd: Optional[Path] = None) -> None:
    cwd = cwd or Path.cwd()
    # Find the pyhqiv root in common clone layouts
    candidates = [
        cwd,
        cwd / "pyhqiv",
        cwd.parent / "pyhqiv",
        cwd / ".." / "pyhqiv",
    ]
    py_root = None
    for c in candidates:
        if (c / "pyproject.toml").exists() and (c / "src" / "pyhqiv").exists():
            py_root = c.resolve()
            break
    if py_root is None:
        py_root = cwd if (cwd / "pyproject.toml").exists() else cwd / "pyhqiv"

    print("=== HQIV Arena Local Run ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(py_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # 1. Alignment (fast gate)
    print("\n-- Stage: Alignment --")
    align_script = py_root / "scripts" / "validate_hqiv_alignment.py"
    if align_script.exists():
        try:
            run([sys.executable, str(align_script), "--verbose"], cwd=py_root, check=False)
        except Exception as e:
            print(f"Alignment script error: {e}")
    else:
        print("(alignment script not found in this tree; using in-package validation)")

    # 2. Scoring
    print("\n-- Stage: Scoring (sigma everywhere) --")
    score_script = py_root / "scripts" / "arena" / "compute_score.py"
    if score_script.exists():
        try:
            run([sys.executable, str(score_script), "--print-badges"], cwd=py_root, check=False)
        except Exception as e:
            print(f"Scoring error: {e}")
    else:
        # Fallback to in-process (always available after editable install or PYTHONPATH)
        try:
            sys.path.insert(0, str(py_root / "src"))
            from pyhqiv.arena import compute_score  # type: ignore

            res = compute_score()
            print(f"overall_score: {res.overall_score}")
            print(f"sigma_weighted: {res.sigma_weighted}")
            print(f"protected regressions: {res.num_regressed_protected}")
            print("Local score computed (in-process).")
        except Exception as e:
            print(f"Could not compute score: {e}")

    print("\nLocal run complete. For the authoritative score (full Lean cert + remote CI), open a PR.")


# --- Submit ------------------------------------------------------------------

def do_submit(note_file: str, model: str, claimed_score: Optional[float], cwd: Optional[Path] = None) -> None:
    cwd = cwd or Path.cwd()
    if not is_hqiv_workspace(cwd):
        print("Warning: not obviously inside an HQIV arena workspace. Continuing anyway.")

    ensure_clean_worktree(cwd, force=False)

    note_path = Path(note_file)
    if not note_path.exists():
        print(f"Note file not found: {note_file}", file=sys.stderr)
        sys.exit(1)
    note = note_path.read_text(encoding="utf-8")

    if len(note) > 10240:
        print("Note is too long (>10 KiB). Trim it.", file=sys.stderr)
        sys.exit(1)

    if not model or len(model) < 3:
        print("--model is required and should identify the model/agent used (e.g. 'Claude 4 Opus').", file=sys.stderr)
        sys.exit(1)

    cfg = read_config()
    token = get_effective_token(cfg)
    if not token and not has_gh():
        print("No GitHub token configured and no `gh` CLI found. Run `hqiv-arena login` first.", file=sys.stderr)
        sys.exit(1)

    # Create a branch if on main
    branch = f"arena/{os.environ.get('USER', 'solver')}-{int(time.time())}"
    try:
        current = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture=True).stdout.strip()
        if current in ("main", "master"):
            run(["git", "checkout", "-b", branch], cwd=cwd)
    except Exception:
        pass

    # Commit if there are changes (best effort)
    try:
        run(["git", "add", "-A"], cwd=cwd, check=False)
        run(["git", "commit", "-m", f"arena: improvement (model: {model})\n\n{ note[:500] }..."], cwd=cwd, check=False)
    except Exception:
        pass

    # Push
    try:
        run(["git", "push", "-u", "origin", "HEAD"], cwd=cwd)
    except subprocess.CalledProcessError:
        print("Push failed. Check your token has 'repo' scope and remote is set.")
        sys.exit(1)

    # Create PR using gh (preferred) or GitHub API
    pr_title = f"Arena submission: {model} (local score ~{claimed_score or 'see note'})"
    pr_body = f"""## HQIV Arena Submission

**Model / Agent**: {model}
**Claimed local score**: {claimed_score or 'see run output'}

{note}

---
*Submitted via hqiv-arena CLI. Full CI scoring (including Lean certificate) will be posted by the Arena workflow.*
"""

    if has_gh():
        try:
            cp = run(
                ["gh", "pr", "create", "--title", pr_title, "--body", pr_body, "--fill"],
                cwd=cwd,
                capture=True,
                check=False,
            )
            print(cp.stdout or cp.stderr)
            return
        except Exception as e:
            print(f"gh pr create had issues, falling back to API: {e}")

    # Fallback: direct GitHub API PR creation (requires token with repo scope)
    if not token:
        print("Cannot create PR: no token and gh not available / failed.", file=sys.stderr)
        sys.exit(1)

    # We need the repo name and head ref
    try:
        origin = run(["git", "remote", "get-url", "origin"], cwd=cwd, capture=True).stdout.strip()
        # https://github.com/owner/repo.git or git@
        if "github.com" in origin:
            parts = origin.split("github.com/")[-1].replace(".git", "").strip("/")
            owner, repo = parts.split("/", 1)
        else:
            raise RuntimeError("origin not a github url")
        head = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture=True).stdout.strip()
        ref = f"{owner}:{head}"

        # Create PR via API
        url = f"{get_effective_api_base(cfg)}/repos/{owner}/{repo}/pulls"
        payload = {
            "title": pr_title,
            "head": head if "/" not in head else ref.split(":", 1)[1],
            "base": "main",
            "body": pr_body,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            pr = json.loads(resp.read())
            print(f"PR created: {pr.get('html_url')}")
    except Exception as e:
        print(f"Failed to create PR via API: {e}", file=sys.stderr)
        sys.exit(1)


# --- Submissions / note (lightweight, use gh or API) ------------------------

def do_submissions(all_public: bool = False, cwd: Optional[Path] = None) -> None:
    cwd = cwd or Path.cwd()
    if has_gh():
        label = "hqiv-arena" if not all_public else None
        args = ["gh", "pr", "list", "--limit", "20", "--json", "number,title,author,headRefName,createdAt,url"]
        if label:
            args += ["--label", label]
        cp = run(args, cwd=cwd, capture=True, check=False)
        print(cp.stdout or "No recent arena PRs found via gh.")
        return

    print("Install `gh` (GitHub CLI) for nice submission listing, or implement custom listing here.")
    print("For now, visit https://github.com/HQIV/hqiv-lean/pulls and https://github.com/disregardfiat/pyhqiv/pulls")


def do_note(ref: str, cwd: Optional[Path] = None) -> None:
    if has_gh():
        # Try to show the PR body
        cp = run(["gh", "pr", "view", ref, "--json", "body"], cwd=cwd or Path.cwd(), capture=True, check=False)
        if cp.returncode == 0:
            data = json.loads(cp.stdout)
            print(data.get("body", "(no body)"))
            return
    print(f"Use `gh pr view {ref}` or open the PR in the browser to read the full note.")


# --- Sync / Reset (frontier maintenance) ------------------------------------

def do_sync(cwd: Optional[Path] = None, force: bool = False) -> None:
    cwd = cwd or Path.cwd()
    ensure_clean_worktree(cwd, force=force)
    print("Fetching latest main ...")
    run(["git", "fetch", "origin", "main"], cwd=cwd, check=False)
    run(["git", "checkout", "main"], cwd=cwd, check=False)
    run(["git", "pull", "--ff-only"], cwd=cwd, check=False)
    print("Synced to current main (best promoted baseline).")
    print("Now `cd` into the py side and run `hqiv-arena run` from a fresh improvement branch.")


def do_reset(ref: str, cwd: Optional[Path] = None, force: bool = False) -> None:
    cwd = cwd or Path.cwd()
    ensure_clean_worktree(cwd, force=force)
    print(f"Resetting workspace to {ref} (best-effort; HQIV uses normal git history).")
    run(["git", "fetch", "origin"], cwd=cwd, check=False)
    run(["git", "checkout", ref], cwd=cwd, check=False)
    print("Reset complete. Inspect the commit and continue improving from that point.")


# --- Version / Update / Install Skill ----------------------------------------

def do_version() -> None:
    print(f"{APP_NAME} (dev)")
    # In a real release we would embed __version__


def do_update() -> None:
    print("Update not implemented for the in-tree dev version.")
    print("Pull the latest from the pyhqiv repo and reinstall editable if needed.")


def do_install_skill(target: str = "all") -> None:
    targets = current_skill_install_targets()
    content = load_skill_content()

    chosen = targets if target == "all" else {target: targets[target]}

    for name, dest in chosen.items():
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        dest.write_text(content, encoding="utf-8")
        dest.chmod(0o644)
        print(f"{name}: {dest}")

    print(f"Installed {SKILL_NAME} skill. Restart your agent app.")


# --- Main --------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="HQIV Arena solver CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # login
    p = sub.add_parser("login", help="Store GitHub PAT for Arena operations")
    p.add_argument("token", nargs="?", help="GitHub PAT (ghp_... or github_pat_...)")
    p.add_argument("--api", help="Override API base (rarely needed)")
    p.set_defaults(func=lambda a: do_login(a.token, a.api))

    # config
    p = sub.add_parser("config", help="Show current configuration")
    p.set_defaults(func=lambda a: print(json.dumps(read_config().to_dict(), indent=2)))

    # benchmark
    p = sub.add_parser("benchmark", help="Show the fixed HQIV benchmark")
    p.set_defaults(func=lambda a: print("HQIV Physics Improvement Arena (hiv-lean + pyhqiv)\nSee https://disregardfiat.tech/#arena"))

    # clone
    p = sub.add_parser("clone", help="Clone a fresh HQIV arena workspace")
    p.add_argument("dir", nargs="?", help="Target directory")
    p.set_defaults(func=lambda a: do_clone(a.dir))

    # setup
    p = sub.add_parser("setup", help="Install dependencies for the current workspace")
    p.set_defaults(func=lambda a: do_setup())

    # run
    p = sub.add_parser("run", help="Run local HQIV Arena scoring / benchmark")
    p.set_defaults(func=lambda a: do_run())

    # submit
    p = sub.add_parser("submit", help="Submit current changes as an Arena PR")
    p.add_argument("--note-file", required=True, help="Markdown file with detailed progress note")
    p.add_argument("--model", required=True, help="Model or agent used (e.g. 'Claude 4 Opus')")
    p.add_argument("--claimed-score", type=float, help="Optional local score you observed")
    p.set_defaults(func=lambda a: do_submit(a.note_file, a.model, a.claimed_score))

    # submissions
    p = sub.add_parser("submissions", help="List recent submissions / PRs")
    p.add_argument("--all", action="store_true", help="Include all public (not just yours)")
    p.set_defaults(func=lambda a: do_submissions(a.all))

    # note
    p = sub.add_parser("note", help="Print the note for a submission / PR")
    p.add_argument("ref", help="PR number or short SHA prefix")
    p.set_defaults(func=lambda a: do_note(a.ref))

    # sync
    p = sub.add_parser("sync", help="Sync workspace to the current best on main")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: do_sync(force=a.force))

    # reset
    p = sub.add_parser("reset", help="Reset to a specific promoted submission")
    p.add_argument("ref")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=lambda a: do_reset(a.ref, force=a.force))

    # version
    p = sub.add_parser("version", help="Show CLI version")
    p.set_defaults(func=lambda a: do_version())

    # update
    p = sub.add_parser("update", help="Update the CLI (if hosted)")
    p.set_defaults(func=lambda a: do_update())

    # install-skill
    p = sub.add_parser("install-skill", help="Install the agent SKILL.md")
    p.add_argument("--target", choices=["all", "agents", "claude", "opencode", "cursor", "grok"], default="all")
    p.set_defaults(func=lambda a: do_install_skill(a.target))

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
