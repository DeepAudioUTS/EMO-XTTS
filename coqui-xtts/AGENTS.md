# Repository Guidelines

## Project Structure & Module Organization
This repository is a minimal Coqui XTTS service rather than a multi-package codebase. Core files live at the repository root:

- `app.py`: FastAPI server that loads the `xtts_v2` model and exposes `GET /` and `POST /tts`.
- `test_xtts.py`: manual smoke-test script for generating one WAV file from terminal input.
- `speaker.wav`: reference speaker sample required for synthesis.
- `output/`: generated audio files written by the API.
- `text.py`: currently unused placeholder; keep utility scripts small and single-purpose if you expand it.

Keep new modules focused and place them at the root unless the project grows enough to justify a `src/` or `tests/` directory.

## Build, Test, and Development Commands
Use a Python virtual environment and install runtime dependencies before running the app.

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install fastapi uvicorn torch TTS`: install the packages used by the current code.
- `uvicorn app:app --reload`: run the API locally with auto-reload.
- `python test_xtts.py`: run the manual synthesis smoke test and write `output.wav`.

If you add new dependencies, document them in a `requirements.txt` or `pyproject.toml`.

## Coding Style & Naming Conventions
Follow standard Python conventions:

- Use 4-space indentation and `snake_case` for functions, variables, and file names.
- Use `PascalCase` for Pydantic models such as `TTSRequest`.
- Keep request validation and error handling explicit at the API boundary.
- Prefer small helpers over large inline blocks when adding synthesis or file-management logic.

No formatter or linter is configured yet. If you add one, prefer `ruff` for linting and formatting and keep the configuration checked in.

## Testing Guidelines
There is no formal `pytest` suite yet. Treat `test_xtts.py` as a manual smoke test for model loading and WAV generation. For new work:

- Add automated tests under a future `tests/` directory using `pytest`.
- Name files `test_<feature>.py`.
- Cover API responses, invalid API keys, empty text handling, and missing `speaker.wav`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so use clear, imperative commit messages such as `Add API key validation for TTS route`. Keep commits scoped to one change. PRs should include:

- A short summary of behavior changes.
- Setup or dependency changes.
- Example request/response details for API changes.
- Sample output paths or logs when audio generation behavior changes.

## Security & Configuration Tips
Do not hardcode secrets in production. Move `API_KEY`, model settings, and file paths from `app.py` into environment variables before deploying.
