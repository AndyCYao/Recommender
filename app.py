from flask import Flask, render_template, request
import recommender

app = Flask(__name__)
rcmdr= recommender.Recommender()

@app.route('/')
def main():
	reader_list = rcmdr.getListOfReaders()
	# print(reader_list)
	return render_template('index.html', 
					       option_list=reader_list)


@app.route("/echo", methods=['POST'])
def echo(): 
    return render_template('index.html', text=request.form['text'])

@app.route("/get_recommender", methods=['POST'])
def get_recommender():
	# print("In get_recommender")
	customerID = request.form['option']
	# print("Test Test {}".format(customerID))
	recommended_list = rcmdr.getRec_Items(customerID)
	return render_template('index.html', tables=[recommended_list])
	# return render_template('index.html', recommended_list=recommended_list)

if __name__ == '__main__':
	app.run(debug = True)