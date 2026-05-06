from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import os, json, base64, cv2
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret123"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---------- INIT ----------
os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

if not os.path.exists("users.json"):
    json.dump([], open("users.json", "w"))

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name","Dept","Date","Time"]).to_csv("attendance.csv", index=False)

# ---------- HELPERS ----------
def load_users():
    return json.load(open("users.json"))

def save_users(data):
    json.dump(data, open("users.json","w"))

# ---------- TRAIN MODEL ----------
def train_model():

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for file in os.listdir("dataset"):
        path = os.path.join("dataset", file)

        name = file.split("_")[0]

        if name not in label_map:
            label_map[name] = current_label
            current_label += 1

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 🔥 IMPROVED PREPROCESSING
        img = cv2.resize(img, (200, 200))
        img = cv2.equalizeHist(img)

        faces.append(img)
        labels.append(label_map[name])

    if len(faces) == 0:
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.array(faces), np.array(labels))
    model.save("trainer/model.yml")

    json.dump(label_map, open("trainer/labels.json","w"))

# ---------- LOGIN ----------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form["username"]
        return redirect("/dashboard")
    return render_template("login.html")

# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", users=load_users())

# ---------- REGISTER ----------
@app.route("/register")
def register():
    return render_template("register.html")

# ---------- SAVE USER ----------
@app.route("/save_user", methods=["POST"])
def save_user():

    name = request.form["name"].strip().lower()
    dept = request.form["dept"]
    regno = request.form["regno"]
    image = request.form["image"]

    users = load_users()

    for u in users:
        if u["username"] == name:
            return jsonify({"status":"exists","msg":"Already Registered"})

    count = len([f for f in os.listdir("dataset") if f.startswith(name)])

    img_data = base64.b64decode(image.split(",")[1])
    path = f"dataset/{name}_{count}.jpg"

    with open(path,"wb") as f:
        f.write(img_data)

    users.append({
        "username": name,
        "dept": dept,
        "regno": regno
    })

    save_users(users)
    train_model()

    return jsonify({"status":"success","msg":"Registered Successfully"})

# ---------- ATTENDANCE ----------
@app.route("/attendance", methods=["POST"])
def attendance():

    name = request.form.get("name")

    if not name or name == "None":
        return jsonify({"msg": "Face not recognized ❌"})

    now = datetime.now()

    df = pd.DataFrame([[name, "Unknown",
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")]],
                      columns=["Name","Dept","Date","Time"])

    df.to_csv("attendance.csv", mode="a", header=False, index=False)

    return jsonify({"msg": f"Attendance marked for {name}"})

@app.route("/take_attendance", methods=["POST"])
def take_attendance():
    return attendance()

@app.route("/attendance_page")
def attendance_page():
    return render_template("attendance.html")

# ---------- RECOGNIZE (🔥 FIXED CORE) ----------
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        img_data = request.form.get("image")
        if not img_data:
            return jsonify({"name": None})

        img_bytes = base64.b64decode(img_data.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
        )

        if len(faces) == 0:
            return jsonify({"name": None})

        if not os.path.exists("trainer/model.yml"):
            return jsonify({"name": None})

        model = cv2.face.LBPHFaceRecognizer_create()
        model.read("trainer/model.yml")

        labels = json.load(open("trainer/labels.json"))
        labels = {v:k for k,v in labels.items()}

        best_name = None
        best_conf = 999

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]

            # 🔥 STRONG FIX (VERY IMPORTANT)
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, (200,200))

            label, conf = model.predict(face)

            print("DEBUG -> Label:", label, "Conf:", conf)

            # 🔥 IMPROVED THRESHOLD
            if conf < best_conf and conf < 85:
                best_conf = conf
                best_name = labels.get(label)

        return jsonify({"name": best_name})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"name": None})

# ---------- DELETE ----------
@app.route("/delete_user", methods=["POST"])
def delete_user():

    name = request.form.get("name")

    users = load_users()
    users = [u for u in users if u["username"] != name]
    save_users(users)

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        df = df[df["Name"] != name]
        df.to_csv("attendance.csv", index=False)

    return jsonify({"status":"success"})

# ---------- HISTORY ----------
@app.route("/history")
def history():

    import csv

    data = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv") as f:
            data = list(csv.reader(f))

    return render_template("history.html", attendance=data)

# ---------- CLEAR ----------
@app.route("/clear_history", methods=["POST"])
def clear_history():
    open("attendance.csv","w").write("Name,Dept,Date,Time\n")
    return jsonify({"status":"success"})

# ---------- EXPORT ----------
@app.route("/export_excel")
def export_excel():
    return send_file("attendance.csv", as_attachment=True)

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------- RUN ----------
if __name__ == "__main__":
    print("🚀 FIXED FACE RECOGNITION SYSTEM RUNNING")
    app.run(debug=True)