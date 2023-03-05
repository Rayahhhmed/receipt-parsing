from flask import Flask, request, jsonify

from ocr import get_item_names_from_receipt
import word_embeddings_ngram as spell_checker
app = Flask(__name__)

@app.route("/upload", methods=["PUT"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."})
    image = request.files["image"]
    image.save("data/" + image.filename)
    res = get_item_names_from_receipt(image.filename)
    return jsonify({"message": res})

@app.before_first_request
@app.route("/recall/add", methods=["PUT"])
def add_to_recalled_list():
    val = spell_checker.add_new_recall_product(request.form.get("product_name"))
    if val == 1:
        return jsonify({"message": "Done"})
    return jsonify({"message": "Recalled product successfully added."})

@app.before_first_request
@app.route("/recall/remove", methods=["DELETE"])
def remove_from_recalled_list():
    spell_checker.remove_recall_product(request.form.get("product_name"))
    return jsonify({"message": "Recalled product successfully removed."})


if __name__ == "__main__":
    app.run(debug=True,port=3002)