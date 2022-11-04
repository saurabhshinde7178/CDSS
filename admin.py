from flask import Flask,redirect,url_for
app=Flask(__name__)

@app.route('/admin')
def admin():
	return "<center><h1>hello adminji</h1></cenetr>"

@app.route('/guest/<guest>')
def guest(guest):
	return "<center><h1>hello %s as guest</h1></cenetr>" % guest

@app.route('/user/<name>')
def user(name):
	if name=="admin":
		return redirect(url_for("admin"))
	else:
		return redirect(url_for("guest",guest=name))

if __name__=="__main__":
	app.run(debug=True)