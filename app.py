import socket
from flask import Flask, render_template, redirect, url_for, request, flash, Response
import flask
from flask_login import LoginManager, login_manager
import flask_login
from flask_login.utils import login_required, login_user, logout_user
from matplotlib.style import available
from sympy import total_degree
from werkzeug.security import check_password_hash
from db import get_booking_collection, get_filled_collection, save_user, get_user, get_admin, store_booking,get_total_parking_spaces, booking_to_filled, remove_from_filled, get_booked_filled_spaces
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import random
import string
from ultralytics import YOLO
import cv2
from flask_socketio import SocketIO , emit
import serial
from subprocess import Popen
import argparse
from flask import jsonify



app = Flask(__name__)
app.secret_key = "secret_key"
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
model = YOLO('E:/PKLot/PKLot/models/fine_tune7.pt')
socketio = SocketIO(app)

label_mapping = {
    'Car': 'Free Space',
    'free': ' Temp Car'
}

occupied_spots_yolo = 0  # Initialize counter for occupied spots
total_spots = get_total_parking_spaces()
occupied_spots_arduino = 0


def gen_frames_from_webcam():
    cap = cv2.VideoCapture(0) # Capture from video file (or use 0 for webcam)
    address = "https://192.168.1.9:8080/video"
    cap.open(address)

    ser = serial.Serial('COM6', 9600)
    global occupied_spots_yolo
    global occupied_spots_arduino
    while True:
        success, frame = cap.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Reset the counter for each frame
            occupied_spots_yolo = 0  

            # Run object detection on the frame
            results = model.predict(frame, conf=0.35)  # Run YOLOv8 prediction
            
            for result in results:
                for label in result.boxes.data[:, 5]:  
                    original_label = result.names[int(label)]
                    # Replace with mapped label
                    new_label = label_mapping.get(original_label, original_label)  # Use original if not found
                    result.names[int(label)] = new_label  # Update the label in results


            for result in results:
                boxes = result.boxes.xyxy  # Bounding box coordinates
                class_ids = result.boxes.cls  # Class IDs
                for class_id in class_ids:
                    if int(class_id) == 1:  # Adjust based on your model's class index for cars
                        occupied_spots_yolo += 1
            
            # Read occupied spots from Arduino
            if ser.in_waiting > 0:
                sensor_data = ser.readline().decode('utf-8').strip()  # Read and decode data
                print(f"Sensor Data: {sensor_data}")  # Debug: print the sensor data
                occupied_spots_arduino = int(sensor_data)
        
            empty_spots = total_spots - occupied_spots_arduino - occupied_spots_yolo
            socketio.emit('update_counts', {'occupied': occupied_spots_yolo + occupied_spots_arduino, 'empty': empty_spots})
            
            # Annotate the frame with detection results
            annotated_frame = results[0].plot()

            # Add text to the annotated frame for total occupied spots
            cv2.putText(annotated_frame, f'Total Occupied: {occupied_spots_yolo}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display total count

            # Encode the frame in JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame as part of a streaming response for Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Exit condition for displaying locally (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows



@app.route('/', methods = ['GET', 'POST'])
def login() :
    message = ''
    if request.method == "POST" :
        username = request.form.get('username')
        password_input = request.form.get('password')
        user = get_user(username)
        
        if user and user.check_password(password_input):
            login_user(user)
            return redirect(url_for('userdashboard'))
        else:
            flash("Username or Password is incorrect.")
    return render_template('index.html', message = message)


@app.route('/userdashboard')
@login_required
def userdashboard():
    total_spaces = get_total_parking_spaces()
    empty_spaces = total_spaces - get_booked_filled_spaces()
    return render_template('userdashboard.html', user = flask_login.current_user, total_spaces = total_spaces, available_spaces = empty_spaces)
  
  
@app.route('/back', methods = ['POST'])
@login_required
def back():
    return redirect(url_for('userdashboard'))


@app.route('/signup')
def signup() :
    return render_template('signup.html')


@app.route('/signup', methods = ['GET', 'POST'])
def registration() :
    message = ''
    if request.method == "POST" :
        username = request.form.get('username')
        password_input = request.form.get('password')
        email = request.form.get('email')
        try :
            save_user(username, email, password_input)
            flash("Successfully Account Created !")
            return redirect(url_for('login'))       
        except DuplicateKeyError:
            message = "User already exists!"
    return render_template('signup.html')
    

@app.route('/logout', methods = ['POST'])
@login_required
def logout():
    logout_user()
    flash("Successfully logged out!")
    return redirect(url_for('login'))

@app.route('/home', methods = ['POST'])
def home() :
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin', methods = ['POST'])
def adminlogin():
    message = ''
    if request.method == "POST" :
        username = request.form.get('username')
        password_input = request.form.get('password')
        admin = get_admin(username)
        if admin and check_password_hash(admin['password'], password_input):
            return redirect(url_for('admindashboard'))
        else :
            message = "Enter valid username and password"
    return render_template('admin.html' , message = message)
    

@app.route('/admindashboard')
def admindashboard() :
    collection1 = get_booking_collection()
    collection2 = get_filled_collection()
    return render_template('admindashboard.html', collection1 = collection1, collection2 = collection2)

@app.route('/video_feed_webcam', methods = ['POST', 'GET'])
def video_feed_webcam():
    
    return Response(gen_frames_from_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/confirm_checked_rows', methods=['POST'])
def confirm_checked_rows():
    data = request.get_json()
    ids_to_remove = data.get('ids', [])
    
    if not ids_to_remove:
        return jsonify(success=False, message="No IDs provided"), 400
    
    try:
        booking_to_filled(ids_to_remove)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route('/remove_checked_rows', methods=['POST'])
def remove_checked_rows():
    data = request.get_json()
    ids_to_remove = data.get('ids', [])
    
    if not ids_to_remove:
        return jsonify(success=False, message="No IDs provided"), 400
    
    try:
        remove_from_filled(ids_to_remove)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500



    
    
   
@app.route('/bookslot')
@login_required 
def bookslot():
    username = flask_login.current_user
    return render_template('booking.html' , user = username)
    
  
@app.route('/bookslot', methods = ['POST'])
@login_required
def conf_booking() :
    if request.method == "POST" :
        customer_name = request.form.get('customer_name')
        arrival_date = request.form.get('arrival_date')
        arrival_time = request.form.get('arrival_time')
        city = request.form.get('city')
        car_number_plate = request.form.get('car_number_plate')
        try:
            # Combine the arrival date and time
            arrival_datetime_str = f"{arrival_date} {arrival_time}"
            arrival_datetime = datetime.strptime(arrival_datetime_str, '%Y-%m-%d %H:%M')

            # Ensure the selected datetime is in the future
            if arrival_datetime <= datetime.now():
                flash('Please select a future date and time.')
                return redirect(url_for('bookslot'))

        except ValueError:
            flash('Invalid date or time format.')
            return redirect(url_for('bookslot'))
        
        parking_slot_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if store_booking(customer_name, arrival_date, arrival_time, city, car_number_plate, parking_slot_id) :
            flash('Booking confirmed successfully! Your Slot ID is: ' + parking_slot_id)
        else :
            flash('No empty spaces are available !')
        
    return redirect(url_for('bookslot'))


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@login_manager.user_loader
def load_user(username) :
    return get_user(username)


if __name__ == "__main__" :
    app.run(debug=True)