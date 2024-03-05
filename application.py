from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import PredictPipeline,extract_features


application=Flask(__name__)

app=application



@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html',final_result=" ")
    else:
        if(request.form.get('url')==""):
            return render_template('index.html',final_result="Enter URL")
    
        data=extract_features(
            request.form.get('url')
        )
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(data)
        if(pred[0]):
            results="This is may be Phishings Website , be aware"
        else:
            results="This 100% safe website"

        return render_template('index.html',final_result=results)
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)