from flask import Flask, render_template
app = Flask(__name__)

@app.route("/hello1/<user>")
def hello1(user):
		return render_template("hello1.html",name=user)

if __name__ == '__main__':
	app.run(debug=True)



