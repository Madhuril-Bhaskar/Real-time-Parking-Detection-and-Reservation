from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from users import User



parkingdb = client.get_database("parkingdb")
users_collection = parkingdb.get_collection("users")
admin_collection = parkingdb.get_collection("admin")
booking_collection = parkingdb.get_collection("booking")
parking_collection = parkingdb.get_collection("parking")
filled_collection = parkingdb.get_collection("filled")



def save_user(username, email, password):
    hash_password = generate_password_hash(password)
    users_collection.insert_one({'_id' : username, 'email' : email, 'password' : hash_password})
    
    
def get_user(username) :
    user_data = users_collection.find_one({'_id' : username})
    return User(user_data['_id'], user_data['email'], user_data['password']) if user_data else None

def save_admin(admin_username, admin_password) :
    hash_password = generate_password_hash(admin_password)
    admin_collection.insert_one({'_id' : admin_username , 'password' : hash_password})

def get_admin(username) :
    admin_data = admin_collection.find_one({'_id' : username})
    return admin_data

def get_total_parking_spaces():
    # Use the aggregate method to sum total parking spaces
    result = parking_collection.aggregate([
        {
            "$group": {
                "_id": None,  # We do not want to group by anything specific
                "total_spaces": { "$sum": "$total_spaces" }
            }
        }
    ])
    
    # Extract the sum from the result
    total_spaces = 0
    for doc in result:
        total_spaces += doc["total_spaces"]
        
    return total_spaces

def store_booking(customer_name, arrival_date, arrival_time, city, car_number_plate, parking_slot_id) :
    total_spaces = get_total_parking_spaces()
    booking_available = booking_collection.count_documents({})
    filled_count = filled_collection.count_documents({})
    if(total_spaces <= booking_available + filled_count) :
        return False
    booking_details = {
    '_customer_name': customer_name,
    'arrival_date': arrival_date,
    'arrival_time': arrival_time,
    'city': city,
    'car_number_plate': car_number_plate,
    'parking_slot_id': parking_slot_id
    }
    booking_collection.insert_one(booking_details)
    return True
    
    
def set_capacity(floor, numberofspot) :
    parking_details = parking_collection.find_one({'_id' : floor})
    if parking_details :
        parking_collection.update_one({'_id' : floor}, {'$set' : {'total_spaces' : numberofspot}})
    else :
        parking_collection.insert_one({'_id' : floor, 'total_spaces' : numberofspot})
        
    
    
def get_booking_collection() :
    collection = list(booking_collection.find())
    return collection


def get_filled_collection() :
    collection = list(filled_collection.find())
    return collection

def booking_to_filled(ids_to_remove) :
    for customer_name in ids_to_remove :
        document = booking_collection.find_one_and_delete({"_customer_name" : customer_name})
        
        if document : 
            filled_collection.insert_one(document)
            
            
def remove_from_filled(ids_to_remove) :
    for customer_name in ids_to_remove :
        filled_collection.find_one_and_delete({"_customer_name" : customer_name})
        

def get_booked_filled_spaces() :
    filled_count = filled_collection.count_documents({})
    booked_count = booking_collection.count_documents({})
    return filled_count + booked_count
