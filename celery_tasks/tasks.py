import celery
from celery import shared_task
from inference import inference
# import os


@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='cattle:get_cattle_task')
def get_cattle_task(self, side_path, rear_path):
    data: dict = {} 
    res = inference.predict(side_path, rear_path)
    # os.remove(side_path)
    # os.remove(rear_path)
    data.update(res)
    data.update({"cattle_id":celery.current_task.request.id})
    return data

