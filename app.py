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

@app.route("/TestJquery", methods=['POST'])
def get_input():
	print("In TestQuery")
	# data = request.
	# return "You Selected {}".format(data)
	pass 

@app.route("/get_recommender", methods=['POST'])
def get_recommender():
	customerID = request.data
	print("customer ID is {} type is {}".format(customerID, type(customerID)))
	if(customerID != ''):
		recommended_list = rcmdr.getRec_Items(customerID)
		return recommended_list
		#return render_template('index.html', tables=[recommended_list])
	else:
		return "Please Try Again"
		#return render_template('index.html', message = "Please Try Again")

if __name__ == '__main__':
	app.run(debug = True)