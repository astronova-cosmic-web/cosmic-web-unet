# Team guide: how our GitHub works and how to submit your work

Welcome! This guide explains how our GitHub is set up and the steps to follow every time you work on a task.
If you only read one section, read **[4. Everyday workflow](#4-everyday-workflow)**.

---

## 1. How everything is organised

| What | Where | What it's for |
|---|---|---|
| **Organisation** | `github.com/astronova-cosmic-web` | Our team's home on GitHub |
| **Repository** | `astronova-cosmic-web/cosmic-web-unet` | All our code, configs, docs and reports |
| **Issues** | Repo → **Issues** tab | One issue per task in our 52-week plan, e.g. `[1.2] Forward model…` |
| **Project board** | Org → **Projects** → *Cosmic Web — 52-week plan* | Who is doing what, and the status of every task |
| **Milestones** | Repo → Issues → **Milestones** | Our 7 checkpoints (W4, W13, W22, W30, W33, W43, W52) |
| **Team** | Org → **Teams** → `core-team` | Gives every member write access |

Every task issue tells you its **phase**, **weeks**, **how many people it needs**, the **output** we expect and when it counts as **done**.

**Labels** show the phase (`phase-1-data`, `phase-3-hybrid-loss`, …) and the type of work (`data`, `ml`, `analysis`, `writing`, `gpu`). `blocked` means "waiting on something".

### Folder layout

```
cosmic-web-unet/
├── src/cosmicweb/
│   ├── data/       forward model (κ → γ₁, γ₂), shape noise, masks, train/val/test splits
│   ├── models/     the U-Net
│   ├── losses/     pixel loss and power-spectrum loss
│   ├── metrics/    RMSE, SSIM, power spectrum, peak counts
│   └── train/      training loop, checkpoints, logging
├── configs/        one YAML file per experiment
├── notebooks/      exploration and figures (keep heavy code in src/)
├── scripts/        command-line helpers
├── tests/          automatic tests (run on every pull request)
├── docs/           notes, reading summaries, data download instructions
├── reports/        figures, report drafts, slides
├── requirements.txt
└── README.md
```

> **Data is never stored on GitHub.** Simulation maps, catalogues and model checkpoints go on our shared drive. Put the link in the issue.

---

## 2. One-time setup (about 20 minutes)

### a) Accept the invitation
Check your email, or **github.com/notifications**, for the invite to `astronova-cosmic-web` and accept it.

### b) Install the tools (Windows)
1. **Git**, which includes Git Bash: git-scm.com → Download → install with the default options.
2. **Python 3.12**: python.org → Downloads. Tick **"Add python.exe to PATH"** on the first screen. It's fine if you also have a newer Python installed.
3. **VS Code** (recommended): code.visualstudio.com, then add the **Python** and **Jupyter** extensions.

On a Mac, Git is already installed and you can get Python 3.12 from python.org.

### c) Tell Git who you are
Open **Git Bash** and run these, using your own name and the email address on your GitHub account:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### d) Download the repo and set up Python
```bash
cd ~/Documents
git clone https://github.com/astronova-cosmic-web/cosmic-web-unet.git
cd cosmic-web-unet

py -3.12 -m venv .venv            # Mac: python3.12 -m venv .venv
source .venv/Scripts/activate     # Mac: source .venv/bin/activate

pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest
```
If the last line says **`1 passed`**, you're ready.

From now on, every time you open a new Git Bash window to work, go into the folder and activate the environment:
```bash
cd ~/Documents/cosmic-web-unet
source .venv/Scripts/activate
```

---

## 3. Picking up a task

1. Open the **project board** and look at tasks for the current weeks.
2. Open the issue and click **Assignees → assign yourself**. If the task needs 2 people, both of you assign yourselves.
3. On the board, move the card from **Todo** to **In progress**.
4. If something is unclear, write a comment on the issue. That keeps the discussion in one place.

---

## 4. Everyday workflow

We **never** work directly on `main`. Every piece of work goes:
**branch → commits → pull request → review → merge.**
GitHub enforces this: you can't push straight to `main`.

### Step 1: Get the latest version
```bash
git checkout main
git pull
```

### Step 2: Create a branch for your task
Name it `feature/<task-id>-short-name`:
```bash
git checkout -b feature/1.2-forward-model
```

### Step 3: Do the work, saving progress as you go
Commit small pieces often, at least once per working session:
```bash
git status                         # see what changed
git add src/cosmicweb/data/forward_model.py tests/test_forward_model.py
git commit -m "Add flat-sky forward model (#6)"
```
- Write commit messages as an instruction: "Add…", "Fix…", "Update…".
- `#6` is the **issue number**, the number GitHub shows next to the issue title. It is **not** the task ID.
- Use `git add <file>` for specific files rather than `git add .`, so you don't accidentally add data files.

### Step 4: Run the tests
```bash
pytest
```

### Step 5: Upload your branch
```bash
git push -u origin feature/1.2-forward-model
```
After the first push, plain `git push` is enough.

### Step 6: Open a pull request (PR)
1. Go to the repo on GitHub. A yellow banner shows **"Compare & pull request"**; click it.
2. **Title:** the same as the task, e.g. `[1.2] Forward model: convergence → shear`.
3. **Description:** fill in the template. Write **`Closes #6`** so the issue closes automatically when the PR is merged.
4. **Reviewers:** choose at least one teammate. Pick someone working on the same phase where possible.
5. Click **Create pull request**, then move the board card to **In review**.

### Step 7: Review and fix
- The **tests** check runs automatically. Wait for the green ✓; if it's a red ✗, click **Details** to see what failed.
- Your reviewer may leave comments. Fix them on the **same branch**, then `git add`, `git commit` and `git push`. The PR updates by itself.

### Step 8: Merge
Once you have **1 approval** and a **green ✓**:
1. Click **Squash and merge**, then **Delete branch**.
2. The issue closes automatically and the card moves to **Done**.

Then tidy up your laptop:
```bash
git checkout main
git pull
git branch -d feature/1.2-forward-model
```

---

## 5. Reviewing a teammate's pull request

When you're asked to review:
1. Open the PR → **Files changed**.
2. Check four things:
   - Does it do what the issue asks?
   - Is it understandable?
   - Are there tests?
   - Are there any data or secret files?
3. Leave comments on specific lines (click the **+** next to a line).
4. Click **Review changes** and choose:
   - **Approve** if it's good to merge;
   - **Comment** for small notes;
   - **Request changes** if something must be fixed first.

Try to review within **2 days**, since your teammate is waiting on you.

---

## 6. Non-code work (reading, writing, figures)

Everything still goes through the same branch → PR flow, so it's reviewed and saved in one place.

| Work | Where it goes |
|---|---|
| Literature notes, reading summaries | `docs/` as Markdown files, e.g. `docs/literature/deepmass-2020.md` |
| Data download instructions | `docs/data.md` |
| Figures for reports and slides | `reports/figures/` (PNG or PDF, keep them small) |
| Report chapters | Overleaf. Put the Overleaf link in the issue and close it with a short comment when done. |
| Experiment results | Summary tables in `reports/`; big files on the shared drive, linked in the issue |

---

## 7. Training on Colab or Kaggle

Write and commit code on your laptop, and use Colab or Kaggle only to **run** it.

```python
# First cell in Colab / Kaggle
!git clone https://github.com/astronova-cosmic-web/cosmic-web-unet.git
%cd cosmic-web-unet
!git checkout feature/2.5-baseline-training   # the branch you want to run
!pip install -q -e . astropy scikit-image pyyaml
# Do NOT reinstall torch or numpy on Colab/Kaggle: use the preinstalled versions.
```

- Save checkpoints and outputs to the **shared drive**, never to GitHub.
- Write the GPU time you used in the issue, e.g. "used ~3 GPU-hours on Kaggle". That helps us track our GPU budget.

---

## 8. Golden rules

1. **Never work on `main`.** Always use a branch.
2. **Never commit data, checkpoints or passwords/API keys.** If you do by accident, tell the team straight away.
3. **One branch per task**, named `feature/<task-id>-name`.
4. **Commit small and often**, with clear messages.
5. **Run `pytest` before opening a PR.**
6. **Keep your issue up to date**: comment on progress, and use the `blocked` label if you're stuck.
7. **Before the weekly advisor meeting (Tuesday):** update your board cards and add a one-line progress comment on your issues.

---

## 9. When things go wrong

| Problem | Fix |
|---|---|
| `push rejected` / `protected branch` | You're on `main`. Create a branch (`git checkout -b feature/...`) and push that. |
| I made changes on `main` by mistake | Run `git checkout -b feature/<id>-name`. Your changes come with you to the new branch. |
| `Your branch is behind` / merge conflict | `git checkout main && git pull && git checkout <your-branch> && git merge main`. Edit the conflicting files (look for `<<<<<<<`), then `git add`, `git commit`, `git push`. Ask for help if unsure. |
| Red ✗ on the PR | Click **Details** to see the error, fix it, then commit and push again. |
| `pytest` can't find `cosmicweb` | Activate the environment (`source .venv/Scripts/activate`), then run `pip install -e .` |
| I committed a data file | Don't push. Ask in the group chat and we'll help you remove it. |

---

## 10. Cheat sheet

```bash
# start of every session
cd ~/Documents/cosmic-web-unet && source .venv/Scripts/activate
git checkout main && git pull

# new task
git checkout -b feature/<task-id>-short-name

# save work
git status
git add <files>
git commit -m "Add/Fix/Update ... (#issue-number)"
pytest
git push -u origin feature/<task-id>-short-name     # then open a PR on GitHub

# after merge
git checkout main && git pull && git branch -d feature/<task-id>-short-name
```

Questions? Ask in the team chat or comment on the issue.
