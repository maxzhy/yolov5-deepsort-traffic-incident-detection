from flask import session, redirect, url_for, request, render_template, jsonify
from controller.modules.user import user_blu

stopflag = True

# Login
@user_blu.route("/login", methods=["GET", "POST"])
def login():
    username = session.get("username")

    if username:
        #return redirect(url_for("home.index"))
        return redirect(url_for("user.startpage"))

    if request.method == "GET":
        return render_template("login.html")
    # obtain parameter
    username = request.form.get("username")
    password = request.form.get("password")
    # check parameter
    if not all([username, password]):
        return render_template("login.html", errmsg="Lack of Input")

    # validate username and password
    if username == "aaa" and password == "aaa":
        # pass
        session["username"] = username
        #return redirect(url_for("home.index"))
        return redirect(url_for("user.startpage"))

    return render_template("login.html", errmsg="Incorrect Username or Password")

# Startpage
@user_blu.route("/startpage")
def startpage():
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))

    print("startpage")
    print("stopflag = true")
    global stopflag
    stopflag = True

    return render_template("startpage.html")

# Logout
@user_blu.route("/logout")
def logout():
    # Delete session
    print("stopflag = true")
    global stopflag
    stopflag = True
    session.pop("username", None)
    # Back to login page
    return redirect(url_for("user.login"))

# Video status
@user_blu.route('/video_status', methods=['POST'])
def video_status():
    username = session.get("username")
    if not username:
        return redirect(url_for("user.login"))

    json = request.get_json()
    status = json['status']

    global stopflag

    if status == "false":
        print("stopflag = false")
        stopflag = False
        return jsonify('stopped!')

    else:
        print("stopflag = true")
        stopflag = True
        return jsonify('started!')
