# Installing this package on the lab server
Extract this ZIP into a separate directory. It already contains one
latent-reasoning-study folder. Keep your original experiments intact.
From the new repository root:
```bash
python scripts/verify_artifacts.py
git init
git status --short
git add README.md LICENSE THIRD_PARTY_NOTICES.md .gitignore environments configs experiments third_party scripts docs results
git diff --cached --stat
```
Inspect staged files and resolve the missing artifacts and licence decision.
Then commit and connect the repository URL you create on GitHub:
```bash
git commit -m "Organize thesis pilot experiments and reproduction notes"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
Replace YOUR_REPOSITORY_URL with your actual URL. Do not paste passwords or
tokens into these commands or the repository. This package has not created
a GitHub repository or pushed any commit. A private draft is appropriate
while dependencies, result evidence and licensing are being completed.
