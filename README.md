<!--
BADGES (ONLY FOR PUBLIC REPOS)
<p align="center">
<a href="https://github.com/antodiazcano/template-project/actions/workflows/ci.yml">
  <img src="https://github.com/antodiazcano/template-project/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>
-->

# template-project

## What to do at first

Install the project and its development dependencies with:

    uv sync

This creates the `.venv` virtual environment automatically. To use a specific
Python version, install and pin it before syncing:

    uv python install 3.13
    uv python pin 3.13
    uv sync

Activating the environment is optional because commands can be run through
`uv run`. To activate it manually:

    source .venv/bin/activate

In VS Code, use `Ctrl` + `Shift` + `P` and select **Python: Select
Interpreter** to choose `.venv`.

Finally, change `name` and `description` in `pyproject.toml`.

## Dealing with dependencies

To add a package, just execute:

    uv add package

To remove a package, just execute:

    uv remove package

Each time a package is added or removed, the change is automatically reflected
in `pyproject.toml` and `uv.lock`.

Runtime dependencies belong in `[project].dependencies`. Since this repository
is a template and currently has no runtime dependencies, that list is empty.

Tools used only for development, such as formatters, linters, test runners, and
documentation generators, belong in the `dev` dependency group. Add or remove a
development dependency with:

    uv add --dev package
    uv remove --dev package

The `dev` group is installed by `uv sync` by default. To install only runtime
dependencies, without development tools, run:

    uv sync --no-dev

You can also create additional named groups when useful:

    uv add --group docs package
    uv remove --group docs package

To include an additional group during synchronization, run:

    uv sync --group <name of the group>

Dependency groups are declared in the `[dependency-groups]` section of
`pyproject.toml`.

You can configure Pylint, Mypy, or other development tools in `pyproject.toml`.

## Saving Tokens with LLMs

- Use [rtk](https://github.com/rtk-ai/rtk):

        # 1. Install for your AI tool
        rtk init -g                     # Claude Code / Copilot (default)
        rtk init -g --gemini            # Gemini CLI
        rtk init -g --codex             # Codex (OpenAI)
        rtk init -g --agent cursor      # Cursor
        rtk init --agent windsurf       # Windsurf
        rtk init --agent cline          # Cline / Roo Code
        rtk init --agent kilocode       # Kilo Code
        rtk init --agent antigravity    # Google Antigravity
        
        # 2. Restart your AI tool, then test
        git status  # Automatically rewritten to rtk git status

- Use [Caveman mode](https://github.com/om-patel5/Caveman-Claude).

- Use lightweight models for some specific tasks like context compression. This [video](https://www.youtube.com/watch?v=NoF-YajElIM) explains it.

## Sanity Checks

With the ```Makefile``` you can use

    make help

to list the available targets,

    make install

to install the dependencies,

    make format

to run isort and Black and format the code,

    make lint

to check the source code and tests with isort, Black, Ruff, Bandit, Mypy,
Flake8, Complexipy, and Pylint without modifying any files,

    make test

to run the tests,

    make check

to run linting and tests,

    make clean

to delete "trash" directories like `__pycache__`. Formatting and cleanup remain
explicit operations so that checks do not modify the working tree.

By default, Pylint fails under a mark of 9.5 and complete test coverage is not
required. However, it is good practice to periodically aim for a 10/10 Pylint
score and 100% test coverage. To see which lines are not covered, run:

    uv run pytest --cov-report term-missing

and they will be shown.

## Others

To preview the documentation, run:

    uv run mkdocs serve
