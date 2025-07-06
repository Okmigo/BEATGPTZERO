git filter-repo --path .ssh --path .bash_history --invert-paths --force
git push --force
git remote add origin https://github.com/Okmigo/BEATGPTZERO.git
git push -u origin main --force
gcloud builds submit --tag gcr.io/beatgptzero/beatgptzero-app
gcloud run deploy beatgptzero-api   --image gcr.io/beatgptzero/beatgptzero-app   --platform managed   --region us-east1   --allow-unauthenticated   --port 3000
git add . && git commit -m "Sync latest deployment-ready code" && git push
[main 9f83508] Sync latest deployment-ready code
git pull --rebase origin main
git push
