# Contributing

1. **One branch per task**, named `feature/<task-id>-short-name`, for example `feature/1.2-forward-model`.
2. **Every change goes through a pull request** with at least one approving review.
3. **Commit messages** use the imperative mood and reference the issue, for example `Add flat-sky forward model (#6)`.
4. **Never commit** data, checkpoints, API keys or tokens. The `.gitignore` blocks the common file types.
5. **Run `pytest` before opening a pull request.** CI runs it again on every PR.
