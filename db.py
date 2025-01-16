from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from users import User
from flask import jsonify



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


def get_parking():
    parking_spots = parking_collection.find()
    parking_data = []
    for spot in parking_spots:
        parking_data.append({
            'id': spot['id'],
            'status': spot['status'],
            'type': spot['type']
        })
    
    return jsonify(parking_data) 
        
        
def get_total_parking_spaces():
    total_spaces = parking_collection.count_documents({})
    return total_spaces
    
def is_car_already_booked(car_number_plate):
    """
    Check if a car with the given number plate is already in the booking table.
    """
    existing_booking = booking_collection.find_one({'car_number_plate': car_number_plate})
    return existing_booking is not None


def store_booking(customer_name, arrival_time, parking_id, parking_type, city, car_number_plate) :
    total_spaces = get_total_parking_spaces()
    booking_available = booking_collection.count_documents({})
    filled_count = filled_collection.count_documents({})
    if(total_spaces <= booking_available + filled_count) :
        return False
    booking_details = {
    '_customer_name': customer_name,
    'arrival_time': arrival_time,
    'parking_id': parking_id,
    'parking_type': parking_type,
    'city': city,
    'car_number_plate': car_number_plate,
    }
    
    booking_collection.insert_one(booking_details)
    
    # Update the status of the parking_id in parking_collection
    result = parking_collection.update_one(
        {'id': parking_id},  # Match the parking_id
        {'$set': {'status': 'booked'}}  # Update the status
    )
    
    # Check if the update was successful
    if result.modified_count == 0:
        print(f"Failed to update parking_id {parking_id} in parking_collection.")
    else:
        print(f"Parking ID {parking_id} successfully marked as booked.")

    return True
    
    
def set_capacity(floor, numberofspot) :
    parking_details = parking_collection.find_one({'_id' : floor})
    if parking_details :
        parking_collection.update_one({'_id' : floor}, {'$set' : {'total_spaces' : numberofspot}})
    else :
        parking_collection.insert_one({'_id' : floor, 'total_spaces' : numberofspot})
        
        

def initialize_parking_data():
    """
    Initializes parking spot data with id, status, and type for the database.
    Creates 5 open roof parking spots (OP-1 to OP-5) and 5 inner parking spots (IP-1 to IP-5).
    """
    # Sample data for parking spots (5 open roof and 5 inner parking spots)
    parking_spots = [
        {'id': 'OP-1', 'status': 'empty', 'type': 'open roof'},
        {'id': 'OP-2', 'status': 'empty', 'type': 'open roof'},
        {'id': 'OP-3', 'status': 'empty', 'type': 'open roof'},
        {'id': 'OP-4', 'status': 'empty', 'type': 'open roof'},
        {'id': 'OP-5', 'status': 'empty', 'type': 'open roof'},
        {'id': 'IP-1', 'status': 'empty', 'type': 'inner parking'},
        {'id': 'IP-2', 'status': 'empty', 'type': 'inner parking'},
        {'id': 'IP-3', 'status': 'empty', 'type': 'inner parking'},
        {'id': 'IP-4', 'status': 'empty', 'type': 'inner parking'},
        {'id': 'IP-5', 'status': 'empty', 'type': 'inner parking'}
    ]
    
    # Insert each parking spot data into the collection
    for spot in parking_spots:
        # Check if the spot already exists (by id)
        existing_spot = parking_collection.find_one({'id': spot['id']})
        
        if not existing_spot:
            # Insert the parking spot data if it doesn't exist
            parking_collection.insert_one(spot)
            print(f"Initialized parking spot {spot['id']} with status '{spot['status']}' and type '{spot['type']}'.")
        else:
            # If the spot already exists, print a message
            print(f"Parking spot {spot['id']} already exists. No changes made.")

# Call the function to initialize the data
# initialize_parking_data()
    
    
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
