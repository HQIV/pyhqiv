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
import tempfile
import textwrap
import time
import urllib.request
import urllib.error
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

DEFAULT_GITHUB_API = "https://api.github.com"
DEFAULT_ARENA_API = "https://disregardfiat.tech/api/v1"

CONFIG_ENV_TOKEN = "HQIV_ARENA_TOKEN"
CONFIG_ENV_API = "HQIV_ARENA_API_URL"

MARKER_FILE = ".hqiv-arena"
LOCAL_GIT_CONFIG_KEY = "hqiv-arena.workspace"

# The two repos that make up the HQIV benchmark workspace
HQIV_LEAN_REPO = "https://github.com/HQIV/hqiv-lean.git"
PYHQIV_REPO = "https://github.com/HQIV/pyhqiv.git"
PYHQIV_GITHUB_WEB = "https://github.com/HQIV/pyhqiv"


@dataclass
class Config:
    arena_api_key: Optional[str] = None
    github_token: Optional[str] = None
    arena_api_url: str = DEFAULT_ARENA_API
    github_api_base: str = DEFAULT_GITHUB_API

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arenaApiKey": self.arena_api_key,
            "githubToken": self.github_token,
            "arenaApiUrl": self.arena_api_url,
            "githubApiBase": self.github_api_base,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        legacy = d.get("token")
        arena_key = d.get("arenaApiKey")
        gh_token = d.get("githubToken")
        if legacy and not arena_key and not gh_token:
            if str(legacy).startswith("hqiv_"):
                arena_key = legacy
            else:
                gh_token = legacy
        arena_url = d.get("arenaApiUrl", DEFAULT_ARENA_API)
        gh_api = d.get("githubApiBase", DEFAULT_GITHUB_API)
        legacy_api = d.get("apiBaseUrl")
        if legacy_api:
            if "disregardfiat.tech" in str(legacy_api) or str(legacy_api).rstrip("/").endswith("/api/v1"):
                arena_url = legacy_api
            else:
                gh_api = legacy_api
        return cls(
            arena_api_key=arena_key,
            github_token=gh_token,
            arena_api_url=arena_url,
            github_api_base=gh_api,
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


def _env_token() -> Optional[str]:
    return os.environ.get(CONFIG_ENV_TOKEN)


def is_arena_api_key(token: str) -> bool:
    return token.startswith("hqiv_")


def is_github_pat(token: str) -> bool:
    return token.startswith(("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_"))


def get_arena_api_key(cfg: Config) -> Optional[str]:
    t = _env_token()
    if t and is_arena_api_key(t):
        return t
    return cfg.arena_api_key


def get_github_token(cfg: Config) -> Optional[str]:
    t = _env_token()
    if t and not is_arena_api_key(t):
        return t
    return cfg.github_token


def get_arena_api_url(cfg: Config) -> str:
    return (os.environ.get(CONFIG_ENV_API) or cfg.arena_api_url).rstrip("/")


def get_github_api_base(cfg: Config) -> str:
    return cfg.github_api_base.rstrip("/")


def arena_api_request(
    cfg: Config,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    key = api_key or get_arena_api_key(cfg)
    if not key:
        raise ValueError("no Arena API key (hqiv_…); sign in at https://disregardfiat.tech/#arena")
    url = f"{get_arena_api_url(cfg)}{path}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Arena API HTTP {e.code}: {detail}") from e


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


# --- Login (Arena API key and/or GitHub PAT) ---------------------------------

def do_login(token: Optional[str], api: Optional[str]) -> None:
    cfg = read_config()
    if api:
        if "github.com" in api or "api.github" in api:
            cfg.github_api_base = api.rstrip("/")
        else:
            cfg.arena_api_url = api.rstrip("/")

    if not token:
        print("HQIV Arena supports two credentials (you can run login twice to store both):")
        print()
        print("  A) Arena API key (hqiv_…) — from https://disregardfiat.tech/#arena")
        print("     Sign in with GitHub on the site; copy the one-time hqiv_ key.")
        print("     Used for provisional leaderboard entries via POST /api/v1/submissions.")
        print()
        print("  B) GitHub PAT (ghp_… / github_pat_…) with 'repo' scope — for push + PR.")
        print("     Or use: gh auth login  (then submit can use gh without storing a PAT).")
        print()
        token = input("Paste token (hqiv_… or GitHub PAT): ").strip()

    if not token:
        print("No token provided.", file=sys.stderr)
        sys.exit(1)

    token = token.strip()

    if is_arena_api_key(token):
        cfg.arena_api_key = token
        try:
            me = arena_api_request(cfg, "GET", "/me", api_key=token)
            who = me.get("github") or me.get("label") or me.get("id")
            print(f"Arena API key validated ({who}).")
        except Exception as e:
            print(f"Warning: could not validate Arena API key ({e}). Storing anyway.")
    elif is_github_pat(token) or token:
        cfg.github_token = token
        try:
            if has_gh() and gh_auth_status():
                print("GitHub CLI is authenticated; PAT also stored for API fallback.")
            else:
                req = urllib.request.Request(
                    f"{get_github_api_base(cfg)}/user",
                    headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github+json",
                    },
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    user = json.loads(resp.read())
                    print(f"GitHub token validated for user: {user.get('login')}")
        except Exception as e:
            print(f"Warning: could not fully validate GitHub token ({e}). Storing anyway.")
    else:
        print("Unrecognized token prefix. Expected hqiv_… or ghp_… / github_pat_…", file=sys.stderr)
        sys.exit(1)

    write_config(cfg)
    print(f"Config saved to {config_path()}")


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
    print(f"Next steps:")
    print(f"  cd {py_dir}")
    print(f"  hqiv-arena setup")
    print(f"  hqiv-arena run")
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

    integrity_script = py_root / "scripts" / "check_arena_source_integrity.py"
    if integrity_script.exists():
        print("\n-- Stage: Source integrity (mirror modules) --")
        try:
            run([sys.executable, str(integrity_script), "--verbose"], cwd=py_root, check=True)
        except subprocess.CalledProcessError:
            print("Source integrity gate FAILED.", file=sys.stderr)
            sys.exit(1)

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
            from pyhqiv.arena import build_default_metrics, compute_score  # type: ignore

            res = compute_score()
            print(f"overall_score: {res.overall_score}")
            print(f"sigma_weighted: {res.sigma_weighted}")
            print(f"protected regressions: {res.num_regressed_protected}")
            print("Local score computed (in-process).")
        except Exception as e:
            print(f"Could not compute score: {e}")

    print("\nLocal run complete. For the authoritative score (full Lean cert + remote CI), open a PR.")


# --- Submit ------------------------------------------------------------------

def _git_sha(cwd: Path) -> Optional[str]:
    try:
        return run(["git", "rev-parse", "HEAD"], cwd=cwd, capture=True).stdout.strip()
    except Exception:
        return None


def post_arena_api_submission(
    cfg: Config,
    *,
    note: str,
    model: str,
    claimed_score: Optional[float],
    sigma_weighted: Optional[float],
    git_ref: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not get_arena_api_key(cfg):
        return None
    body: Dict[str, Any] = {"note": note, "model": model}
    if claimed_score is not None:
        body["claimed_score"] = claimed_score
    if sigma_weighted is not None:
        body["sigma_weighted"] = sigma_weighted
    if git_ref:
        body["git_ref"] = git_ref
    return arena_api_request(cfg, "POST", "/submissions", body)


def do_submit(
    note_file: str,
    model: str,
    claimed_score: Optional[float],
    sigma_weighted: Optional[float] = None,
    pr_only: bool = False,
    api_only: bool = False,
    cwd: Optional[Path] = None,
) -> None:
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
    git_ref = _git_sha(cwd)

    if not pr_only:
        api_result = post_arena_api_submission(
            cfg,
            note=note,
            model=model,
            claimed_score=claimed_score,
            sigma_weighted=sigma_weighted,
            git_ref=git_ref,
        )
        if api_result:
            print("Arena API submission recorded (provisional leaderboard).")
            print(api_result.get("message", ""))
            preview = api_result.get("leaderboard_preview") or {}
            if preview.get("current_best"):
                print(f"  current_best score: {preview['current_best'].get('score')}")
        elif api_only:
            print("No Arena API key. Sign in at https://disregardfiat.tech/#arena", file=sys.stderr)
            sys.exit(1)

    if api_only:
        return

    github_token = get_github_token(cfg)
    if not github_token and not has_gh():
        print(
            "No GitHub PAT or `gh` CLI for PR workflow. Recorded Arena API submission only.\n"
            "For authoritative CI scoring, run: hqiv-arena login ghp_…  (or gh auth login)",
            file=sys.stderr,
        )
        if pr_only:
            sys.exit(1)
        return

    if pr_only and not github_token and not has_gh():
        print("PR workflow requires GitHub PAT or gh auth.", file=sys.stderr)
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
    if not github_token:
        print("Cannot create PR: no GitHub token and gh not available / failed.", file=sys.stderr)
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
        url = f"{get_github_api_base(cfg)}/repos/{owner}/{repo}/pulls"
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
                "Authorization": f"token {github_token}",
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
    cfg = read_config()
    if get_arena_api_key(cfg):
        try:
            path = "/submissions?all=1" if all_public else "/submissions"
            data = arena_api_request(cfg, "GET", path)
            subs = data.get("submissions") or []
            if subs:
                print("=== Arena API submissions ===")
                for s in subs[:20]:
                    print(
                        f"  {s.get('created_at', '')}  {s.get('author', '?')}  "
                        f"model={s.get('model', '?')}  score={s.get('claimed_score')}"
                    )
            else:
                print("No Arena API submissions yet.")
        except Exception as e:
            print(f"Arena API submissions list failed: {e}")

    if has_gh():
        label = "hqiv-arena" if not all_public else None
        args = ["gh", "pr", "list", "--limit", "20", "--json", "number,title,author,headRefName,createdAt,url"]
        if label:
            args += ["--label", label]
        cp = run(args, cwd=cwd, capture=True, check=False)
        print(cp.stdout or "No recent arena PRs found via gh.")
        return

    print("Install `gh` (GitHub CLI) for nice submission listing, or implement custom listing here.")
    print(f"For now, visit https://github.com/HQIV/hqiv-lean/pulls and {PYHQIV_GITHUB_WEB}/pulls")


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
    p = sub.add_parser("login", help="Store Arena API key (hqiv_…) and/or GitHub PAT")
    p.add_argument("token", nargs="?", help="hqiv_… from disregardfiat.tech/#arena or ghp_… PAT")
    p.add_argument(
        "--api",
        help="Override Arena API URL (https://disregardfiat.tech/api/v1) or GitHub API base",
    )
    p.set_defaults(func=lambda a: do_login(a.token, a.api))

    # config
    p = sub.add_parser("config", help="Show current configuration")
    p.set_defaults(
        func=lambda a: print(
            json.dumps(read_config().to_dict(), indent=2)
            + "\n# HQIV_ARENA_TOKEN / HQIV_ARENA_API_URL override file values"
        )
    )

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
    p.add_argument("--sigma-weighted", type=float, help="Optional weighted sigma from local run")
    p.add_argument("--api-only", action="store_true", help="Only POST to disregardfiat Arena API")
    p.add_argument("--pr-only", action="store_true", help="Skip Arena API; only open GitHub PR")
    p.set_defaults(
        func=lambda a: do_submit(
            a.note_file,
            a.model,
            a.claimed_score,
            a.sigma_weighted,
            a.pr_only,
            a.api_only,
        )
    )

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
