# sleep 60
tmux \
    new-session  'eval "$(conda shell.bash hook)" ; conda activate deploy1 ; cd /home/acme_admin/fastapi-celery-rabbitmq-application; gunicorn -k  uvicorn.workers.UvicornWorker main:app  -b 0.0.0.0:9001 ; bash' \; \
    split-window 'eval "$(conda shell.bash hook)" ; conda activate deploy1 ; cd /home/acme_admin/fastapi-celery-rabbitmq-application; celery -A main.celery worker --loglevel=info -Q cattle --concurrency=2 ; bash'


# new-session  'eval "$(conda shell.bash hook)" ; conda activate deploy1 ; celery -A main.celery worker --loglevel=info -Q cattle --concurrency=2 ; bash' \; \
# split-window 'eval "$(conda shell.bash hook)" ; conda activate deploy1 ; gunicorn -k  uvicorn.workers.UvicornWorker main:app  -b 0.0.0.0:9001 ; bash' 
