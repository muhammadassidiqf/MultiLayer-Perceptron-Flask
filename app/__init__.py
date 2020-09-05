from flask import Flask

app = Flask(__name__) #untuk menjelaskan nama modul yang digunakan agar folder lain memanggil folder app akan otomatis

from app import routes #Memanggil file routes 