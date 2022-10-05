from bs4 import BeautifulSoup
import requests, datetime 
import sqlite3

MANSONNET = "https://markmanson.net"
DATABASE_NAME = "articles.db"