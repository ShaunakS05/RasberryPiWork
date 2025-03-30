from pymongo import MongoClient
from datetime import datetime
import uuid
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient('mongodb+srv://nyanprak:Samprakash3!@trash.utmo5ml.mongodb.net/?retryWrites=true&w=majority&appName=trash')
db = client['trash_management_db']
collection = db['trash_cans']

# Generate a unique ID for the new item
new_item_id = f"item-{uuid.uuid4().hex[:8]}"

# Create the new item
new_item = {
    "id": new_item_id,
    "type": "landfill",  # or "recycle", "compost", etc.
    "name": "wrwetwrty",  # name of the trash item
    "timestamp": datetime.now().isoformat()
}

# Add the item to the array using the $push operator
trash_can_id = "67e90d1dc1ede39d902e351a"  # The ObjectId from your example
result = collection.update_one(
    {"_id": ObjectId(trash_can_id)},  # Convert string to ObjectId
    {"$push": {"items": new_item}}
)

# Also update the activity history for today
today = datetime.now().strftime("%m/%d")
result = collection.update_one(
    {"_id": ObjectId(trash_can_id), "activityHistory.date": today},
    {"$inc": {"activityHistory.$.items": 1}}
)

# If today isn't in the activity history yet, add it
if result.modified_count == 0:
    collection.update_one(
        {"_id": ObjectId(trash_can_id)},
        {"$push": {"activityHistory": {"date": today, "items": 1}}}
    )

print(f"Added item {new_item_id} to the trash can")