from flask import Flask, redirect, url_for, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import cv2
import numpy as np
import re
import json
import urllib
import pandas as pd
import requests as req
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class Shopee:
    
    def __init__(self, Search = "ban"):
        
        self.ENDPOINT = f"https://shopee.co.id/api/v4/search/search_items?by=relevancy&keyword={Search}&limit=100&newest=100&order=desc&page_type=search&scenario=PAGE_GLOBAL_SEARCH&version=2"
        self.IMAGE_ENDPOINT = "https://cf.shopee.co.id/file/"
        self.SHOP = "https://shopee.co.id/api/v4/product/get_shop_info?shopid="
        
        
    def getShopee(self) :
        
        sorting          = []
        request_products = json.loads(req.get(self.ENDPOINT).content)
        
        for indexProducts in range(len(request_products["items"])) :
            
            price        = str(request_products["items"][indexProducts]["item_basic"]["price"])
            
            sorting.append(
            
                int(price[0:len(price)-5])
            
            )
            
        return sorting

class ecommerceData:
    
    def SORT(self, Search):   
        shopee = Shopee(Search).getShopee()
        shopee.sort()
        return "Range prediksi harga : " + str(shopee[0]) + " - " + str(shopee[len(shopee)-1])

app = Flask(__name__)
CORS(app) #c

@app.route("/")
def hello_world():
    return "<p>Welcome to TrashEdu AI !!</p>"

@app.route("/predict_sampah", methods = ['POST'])
def fromform():
    if 'file' not in request.files:
        resp =  {
                 'status': 400,
                 'error': 'No selected file'
                }
        return jsonify(resp)

    if request.files['file'].filename == '':
        resp =  {
                 'status': 400,
                 'error': 'No selected file'
                }
        return jsonify(resp)
        
    output_class = ["batteries",
                "biological",
                "brown glass",
                "cardboard",
                "clothes", 
                "e waste",
                "glass",
                "green glass",
                "light blubs", 
                "metal", 
                "organic", 
                "paper", 
                "plastic",
                "shoes",
                "trash",
                "white glass"]
    lama_waktu = [
        "100 tahun",
        "Sampah Berbahaya mencemari lingkungan !!",
        "1 juta tahun",
        "6 bulan",
        "2-20 tahun",
        "1 juta tahun",
        "1 juta tahun",
        "1 juta tahun",
        "1 juta tahun",
        "200 tahun",
        "6 bulan",
        "2-5 bulan",
        "50-100 tahun",
        "20 tahun",
        "Tergantung",
        "Tidak dapat "
    ]
    cara_olah = [
        "Daur ulang, terdapat tempat pengolahan khusus",
        "Terdapat tempat pengolahan khusus",
        "Daur ulang",
        "Daur ulang, penggunaan ulang",
        "Daur ulang",
        "Terdapat beberapa kandungan Elektronik-Waste yang bersifat toxic (membahayakan)",
        "Daur ulang",
        "Daur ulang",
        "Daur ulang",
        "Daur ulang dengan peleburan",
        "Daur ulang lingkungan.",
        "Daur ulang, penggunaan ulang",
        "Daur ulang, penggunaan ulang",
        "Daur ulang, penggunaan ulang",
        "Terdapat berbagai jenis sampah sesuai kategorinya",
        "Daur ulang"
    ]
    Catatan_Tambahan = [
        "Pada baterai terdapat timbal sehingga berbahaya jika dilakukan penguraiaan di tanah",
        "Sampah yang ada mungkin saja mengandung kuman, terdapat tempat pengolahan khusus utnuk menangani sampah medis, silahkan hubungi pihak terkait",
        "Gelas bisa saja terbuat dari kaca, berhati hati Ketika memegang sampah jenis ini, bisa saja melukai diri anda",
        "Sampah tergolong ramah lingkungan, mudah di daur ulang oleh lingkungan.",
        "Terdapat beberapa kain yang ramah lingkungan seperti : Kain linen, katun organic, kain Tencel, kain hemp, kain serat bambu. Jenis kain ramah lingkungan akan lebih mudah terurai dan dikomposkan. Sedangkan pada jenis bahan kain sintetis ternyata memakan waktu yang lama untuk dapat terurai.",
        "E-waste termasuk limbah B3 (bahan berbahaya dan beracun) yang tidak bisa dibuang dan dikelola sembarangan. Banyak zat berbahaya yang bisa mengkontaminasi tubuh dan ekosistem lingkungan, misalnya kandungan mercury dan palladium yang sifatnya beracun.",
        "Gelas bisa saja terbuat dari kaca, berhati hati Ketika memegang sampah jenis ini, bisa saja melukai diri anda",
        "Gelas bisa saja terbuat dari kaca, berhati hati Ketika memegang sampah jenis ini, bisa saja melukai diri anda",
        "Gelas bisa saja terbuat dari kaca, berhati hati Ketika memegang sampah jenis ini, bisa saja melukai diri anda",
        "limbah logam berat ini ke perairan dapat mengurangi kualitas air. Selain itu, logam berat yang terendapkan bersama dengan sedimen juga dapat menyebabkan transfer bahan kimia beracun dari sedimen ke organisme yang ada di perairan tersebut.",
        "Sampah organik yang nantinya membusuk di dalam tanah dapat memberikan unsur hara yang dapat membuat tanah menjadi subur dan tanaman tumbuh lebih sehat.",
        "Sampah tergolong ramah lingkungan, mudah di daur ulang oleh lingkungan.",
        "Sampah kantong plastik dapat mencemari tanah, air, laut, bahkan udara. Kantong plastik terbuat dari penyulingan gas dan minyak yang disebut ethylene. Minyak, gas dan batu bara mentah ",
        "Gambar yang di berikan tidak dapat dimengerti oleh sistemGambar yang di berikan tidak dapat dimengerti oleh sistem",
        "Gelas bisa saja terbuat dari kaca, berhati hati Ketika memegang sampah jenis ini, bisa saja melukai diri anda"
    ]

    img = cv2.cvtColor(cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

    test_image = np.expand_dims(resized, axis=0)

    model = keras.models.load_model('classifyWastetesting2.h5')
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_value_waktu = lama_waktu[np.argmax(predicted_array)]
    predicted_cara_olah = cara_olah[np.argmax(predicted_array)]
    predicted_catatan = Catatan_Tambahan[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    resp =  {
                 'status': 200,
                 'success': "Your waste material is " + str(predicted_value) + " with " +  str(predicted_accuracy) + " % accuracy",
                 'data':{
                     'tipe': str(predicted_value),
                     'lama waktu terurai': str(predicted_value_waktu),
                     'Cara Pengelolahan sampah': str(predicted_cara_olah),
                     'Catatan Tambahan': str(predicted_catatan)
                 }
                }
    return jsonify(resp)

@app.route("/range_harga", methods = ['POST'])
def range_harga():
    if 'word' not in request.form:
        resp =  {
                 'status': 400,
                 'error': 'No selected word'
                }
        return jsonify(resp)

    if request.form['word'] == '':
        resp =  {
                 'status': 400,
                 'error': 'No selected word'
                }
        return jsonify(resp)

    stri = ecommerceData().SORT(
        Search = request.form['word'] 
    )
    resp =  {
            'status': 200,
            'success': stri
            }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True)