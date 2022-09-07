from fastapi import APIRouter,File, UploadFile
from starlette.responses import JSONResponse
import os
import uuid

from celery_tasks.tasks import get_cattle_task 
from config.celery_utils import get_task_info




router = APIRouter(prefix='/cattle', tags=['Cattle'], responses={404: {"description": "Not found"}})


image = 'images'

def save_file(filename, data):
    try:
        os.makedirs(image, exist_ok=True)
    except OSError as error:
        print(error)
        pass

    with open(image+'/'+filename, 'wb') as f:
        f.write(data)



@router.post("/async")
async def get_cattle_async(side: UploadFile = File(...), rear: UploadFile = File(...)):
 
    Side = await side.read()
    Rear = await rear.read()

    cattle_id = str(uuid.uuid4())
    Side_name = cattle_id+side.filename
    Rear_name = cattle_id+rear.filename

    # contents = await myfile.read()

    save_file(Side_name, Side)
    save_file(Rear_name, Rear)

    side_path = image+'/'+Side_name
    rear_path = image+'/'+Rear_name
    task = get_cattle_task.apply_async((side_path,rear_path))
    return JSONResponse({"task_id": task.id})


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """
    Return the status of the submitted Task
    """
    return get_task_info(task_id)

