from flask import Flask, render_template, request
import recommender

app = Flask(__name__)
rcmdr= recommender.Recommender()

@app.route('/')
def main():
	reader_list = rcmdr.getListOfReaders()
	print(reader_list)
	return render_template('index.html', 
					       option_list=reader_list)


@app.route("/echo", methods=['POST'])
def echo(): 
    return render_template('index.html', text=request.form['text'])

if __name__ == '__main__':
	app.run(debug = True)