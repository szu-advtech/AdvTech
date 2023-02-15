from flask import Flask,request
from flask_cors import cross_origin
import json
from solver import Solver

app = Flask(__name__)

@app.route('/solve', methods=['POST','GET'])
@cross_origin()
def solve():
	data = request.get_json(silent=True)
	if "elements" in data:
		solver = Solver(data["elements"])
		result_list = solver.solve()
		return json.dumps(result_list).encode('utf-8')
	return "failed, no result."

if __name__ == '__main__':
    app.run()
