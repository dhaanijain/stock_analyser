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
