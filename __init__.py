from flask import Flask

app = Flask(__name__)

@app.route('/hello/<name>')
def index(name):
	return "<b>Hello World from %s</b>" % name

if __name__ == "__main__":
	app.run(debug=True)
