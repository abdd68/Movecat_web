#/bin/bash

git add . 
git commit -m "a" 
git push heroku main
heroku ps:scale web=1
heroku open

# database:redis
heroku addons:create heroku-redis:mini --app lymph-detection
heroku redis:cli

heroku addons:destroy redis-dimensional-79467 --app lymph-detection --confirm lymph-detection