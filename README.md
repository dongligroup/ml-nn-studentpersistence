# Student Persistence Full-Stack Intelligent App

## Workflow for the `main` branch

> to maintain a clean `main` branch, a simple protection rule has been set to "require one review".

### 1. Pull the Latest Changes

```bash
git pull origin main
```

### 2. Create a Feature Branch

```bash
git checkout -b <feature-branch-name>
```

### 3. Make Changes and Commit

```bash
git add .
git commit -m "Your descriptive commit message"
```

### 4. Keep Updating Your Branch

```bash
git fetch origin
git merge origin/main
```

### 5. Push Your Branch

```bash
git push origin <feature-branch-name>
```

### 6. Create a Pull Request (PR)

Go to tab <kbd>Pull Requests</kbd> on GitHub.
Click New Pull Request, set the source as `<your branch>` and the target as `main`.
Add a title and description, and request at least one reviewer.

### 7. Address Feedback

- If reviewers request changes, modify and push again.
- Once approved, the reviewer can merge the PR into main on GitHub.
- Optionally, delete the feature branch.

> **Tips:**
>
> - Always pull the latest changes from `main` before starting new work to avoid conflicts.
> - If you have merge conflicts, resolve them locally and push the changes back to the feature branch before updating the PR.

## Functionality & Endpoints

| Endpoint      | Method | Description                                           |
|---------------|--------|-------------------------------------------------------|
| `/predict`    | POST   | Accepts a single data point and returns the prediction. |
| `/train-one`  | POST   | Accepts a single labeled data point and trains the model. |
| `/train-batch`| POST   | Accepts a CSV file containing labeled data and trains the model. |

## Sequence Diagram

![alt Sequence Diagram](/images/sequence.png)
