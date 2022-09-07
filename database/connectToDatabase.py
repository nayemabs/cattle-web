from tortoise import Tortoise
async def connectToDatabase():
    await Tortoise.init(
        db_url='postgres://kgtxlqvibkenej:6f7912e6fcddf1fada6c34387943f26b38c301e709d25c093c4aa5fc1f85359e@ec2-44-207-126-176.compute-1.amazonaws.com:5432/dci6siaaqphto',
        modules={'models': ['model.models']}
    )