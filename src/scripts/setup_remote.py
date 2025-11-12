import logging
import os
from argparse import ArgumentParser

from fabric import Connection

from aml4cv import BASE_DIR, log, set_logging_level

set_logging_level(logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Setup a remote machine with the project.")

    parser.add_argument(
        "--remote",
        required=True,
        help="The address to the remote machine, for example, user@host",
    )
    parser.add_argument(
        "--repo",
        help="The repository to clone, on the remote machine.",
    )
    parser.add_argument(
        "--repo-path",
        default="~/project",
        help="The path to clone the repository to, on the remote machine.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("GIT_USER"),
        help="Git user name to configure on the remote machine (git config user.name)",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("GIT_EMAIL"),
        help="Git email address to configure on the remote machine "
        "(git config user.email)",
    )
    parser.add_argument(
        "--signing-key",
        default=os.environ.get("GIT_SIGNING_KEY"),
        help="SSH signing key for Git commit signing "
        "(enables gpg.format ssh and commit.gpgsign)",
    )

    return parser.parse_args()


def check_args(args):
    """Validate that all required arguments have values and are valid."""
    errors = []

    # Check required Git configuration
    if not args.user:
        errors.append(
            "Git user name is required. Set GIT_USER environment variable "
            "or use the --user flag."
        )

    if not args.email:
        errors.append(
            "Git email is required. Set GIT_EMAIL environment variable or "
            "use the --email flag."
        )

    if not args.repo:
        errors.append("Git repository is required. Use the --repo flag.")

    # Validate remote format
    if "@" not in args.remote:
        errors.append(
            f"Remote address should be in format 'user@host', got: {args.remote}"
        )

    if errors:
        raise ValueError(f"Validation errors found: {', '.join(errors)}")

    return True


def setup_remote():
    """Setup a remote machine with the project."""
    args = parse_args()

    REMOTE = args.remote
    REPO = args.repo
    REPO_PATH = args.repo_path
    GIT_EMAIL = args.email
    GIT_USER = args.user
    GIT_SIGNING_KEY = args.signing_key

    check_args(args)

    log(f"Setting up remote machine: {REMOTE}", level=logging.INFO)

    with Connection(REMOTE) as c:
        # check if dir exists
        log(f"Cloning repository: {REPO}", level=logging.INFO)
        result = c.run(f"test -d {REPO_PATH}", warn=True)
        if result.return_code == 0:
            log(f"Repository already exists at {REPO_PATH}", level=logging.WARNING)
        else:
            c.run(f"git clone {REPO} {REPO_PATH}")

        # Parse repository path
        if "~" in REPO_PATH:
            REPO_PATH = f"/home/{c.user}/{REPO_PATH.strip('~/')}"

        # Update system dependencies and install GitHub CLI
        log("Installing GitHub CLI...", level=logging.INFO)
        c.sudo("apt install gh -y", hide=True)

        with c.cd(REPO_PATH):
            log("Copying .env file...", level=logging.INFO)
            c.put(local=str(BASE_DIR / ".env"), remote=f"{REPO_PATH}/.env")

            log("Setting up git authentication...", level=logging.INFO)
            c.run(f"git config --global user.email '{GIT_EMAIL}'")
            c.run(f"git config --global user.name '{GIT_USER}'")

            # Setup git signing if key is provided
            if GIT_SIGNING_KEY:
                c.run(f"git config --global user.signingkey {GIT_SIGNING_KEY}")
                c.run(f"git config --global commit.gpgsign true")
                c.run(f"git config --global gpg.format ssh")

            log("Setting up GitHub authentication...", level=logging.INFO)
            c.run("gh auth login -p https -w", pty=True)

            log("Installing the project and its dependencies...", level=logging.INFO)
            c.run("make install")

    log(f"Setup for remote `{REMOTE}` is complete!", level=logging.INFO)


if __name__ == "__main__":
    setup_remote()
