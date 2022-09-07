
import celery
from celery import shared_task
from celery.signals import worker_process_init, worker_process_shutdown

from inference import inference
import os
from model.models import Cattle
# from tortoise.contrib.fastapi import HTTPNotFoundError, register_tortoise
from database.connectToDatabase import connectToDatabase
from model.models import Cattle






# @shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
#              name='db:store_to_db')
# def store_to_db(cattle_dict,b):


#     cattle_id = cattle_dict.get("cattle_id")
#     cattle_weight = cattle_dict.get("weight")
#     cattle_remarks = cattle_dict.get("remarks")
#     # print(type(Cattle.create(cattle_id = cattle_id, weight=cattle_weight, remarks=cattle_remarks))) 


#     return(cattle_remarks)

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5},
             name='cattle:get_cattle_task')
def get_cattle_task(self, side_path, rear_path):
    # print(celery.current_task.request)
    data: dict = {} 
    res = inference.predict(side_path, rear_path)
    os.remove(side_path)
    os.remove(rear_path)
    # print(type(res))
    data.update(res)
    data.update({"cattle_id":celery.current_task.request.id})
    # print(data)
    return data

