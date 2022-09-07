from tortoise.models import Model
from tortoise import fields
class Cattle(Model):

    cattle_id = fields.TextField()
    weight = fields.FloatField()
    remarks = fields.TextField()
   