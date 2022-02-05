from flask import Flask, request

app = Flask(__name__)

@app.route("/", methos=['GET', 'POST'])
def hello_world():

    print(request.form)

    param_hotel = request.form.get("hotel_type")
    param_month = request.form.get("arrival_month")
    param_num = request.form.get("number_of_people")

    import pickle
    import numpy as np

    with open('exported_one_hot.pickle', 'rb') as fp:
        enc = pickle.load(fp)

    with open('exported_classifier.pickle', 'rb') as fp:
        classifier = pickle.load(fp)

    hotel_feature = enc.transform([[param_hotel]]).toarray()
    month_feature = (int(param_month) >= 6) and (int(param_month) <= 8)

    features = np.hstack([hotel_feature, np.array([[month_feature]]), np.array([[param_num]])])

    if classifier.predict(features)[0]:
        return "will not cancel"
    else:
        return "will cancel"

if __name__ == '__main__':
    app.run(port=5000, debug=True)