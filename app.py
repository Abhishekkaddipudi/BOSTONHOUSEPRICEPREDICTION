import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open("boston_house_pred.pkl","rb"))
scalar=pickle.load(open("scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict_api",methods=['POST'])
def predict_api():
    data=request.json['data']
    print(np.array(list(data.values())).reshape(1,-1))
    transform_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(transform_data)
    print(output[0])
    return jsonify(output[0])
@app.route("/predict",methods=["POST"])
def predict():
    data=[0 if isinstance(x,str) else float(x) for x in request.form.values() ]


            

    final_input=scalar.transform(np.array(data).reshape(1,-1))
   
    output=regmodel.predict(final_input)[0]
    return render_template("index.html",prediction_text=output)

if __name__ =="__main__":
    app.run(debug=True)
    

