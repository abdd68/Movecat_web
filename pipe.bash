#/bin/bash

git add .
git commit -m "a"
git push heroku main
heroku ps:scale web=1
heroku open

# database:redis
heroku redis:cli