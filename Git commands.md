# Git Commands Summary

| Command                        | Description                                                      |
|--------------------------------|------------------------------------------------------------------|
| `git init`                     | Initialize a new Git repository                                  |
| `git clone <repo_url>`         | Clone a remote repository to your local machine                  |
| `git status`                   | Show the status of changes as untracked, modified, or staged     |
| `git add <file>`               | Stage a file for the next commit                                 |
| `git add .`                    | Stage all changes in the current directory                       |
| `git commit -m "message"`      | Commit staged changes with a message                             |
| `git log`                      | Show the commit history                                          |
| `git diff`                     | Show changes between commits, commit and working tree, etc.      |
| `git branch`                   | List, create, or delete branches                                 |
| `git checkout <branch>`        | Switch to a different branch                                     |
| `git merge <branch>`           | Merge another branch into the current branch                     |
| `git pull`                     | Fetch and merge changes from the remote repository               |
| `git push`                     | Push local commits to the remote repository                      |
| `git remote -v`                | Show remote connections                                          |
| `git fetch`                    | Download objects and refs from another repository                |
| `git reset --hard <commit>`    | Reset working directory and index to a specific commit           |
| `git rm <file>`                | Remove a file from the working directory and staging area        |
| `git stash`                    | Stash the changes in a dirty working directory                   |
| `git tag`                      | List, create, or delete tags                                     |

## Configuring User Name and Email in Git

To set your global user name:

```powershell
git config --global user.name "Your Name"
```

To set your global email:

```powershell
git config --global user.email "your.email@example.com"
```

Replace `Your Name` and `your.email@example.com` with your actual name and email address.

To set them only for the current repository (not globally), omit the `--global` flag:

```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

To check your current configuration:

```powershell
git config --list
```

## Connecting to a Remote Repository

To add a remote:

```powershell
git remote add origin <remote_repository_url>
```

To change the remote URL:

```powershell
git remote set-url origin <new_remote_repository_url>
```

To verify remotes:

```powershell
git remote -v
```

## Mapping Local master to Remote main

Rename local branch and set upstream:

```powershell
git branch -m master main
git fetch origin
git branch -u origin/main main
git push origin main
git push --set-upstream origin main
```

## Remove Link to Remote

To remove the remote named origin:

```powershell
git remote remove origin
```

## Force Push to Remote

To force push your local branch to the remote (overwriting remote):

```powershell
git push --force origin main
```

## Handling 'fetch first' Push Error

If you see an error about needing to fetch first:

1. Pull remote changes:

    ```powershell
    git pull origin main
    ```

2. Resolve any conflicts, commit, then push:

    ```powershell
    git push origin main
    ```

If you are sure your local branch is correct and want to overwrite remote:

```powershell
git push --force origin main
```
