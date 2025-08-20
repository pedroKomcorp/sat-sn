import sqlite3
import pandas as pd

def icms():
  conn = sqlite3.connect('resources/tributos.db')
  cursor = conn.cursor()
  cursor.execute('select * from icms')
  
  # Fetch all data
  data = cursor.fetchall()
  
  # Get column names from cursor description
  columns = [description[0] for description in cursor.description]
  
  # Create DataFrame with data and column names
  df = pd.DataFrame(data, columns=columns)
  
  conn.close()
  return df
  
def piscofins():
  conn = sqlite3.connect('resources/tributos.db')
  cursor = conn.cursor()
  cursor.execute('select * from piscofins')

  # Fetch all data
  data = cursor.fetchall()

  # Get column names from cursor description
  columns = [description[0] for description in cursor.description]

  # Create DataFrame with data and column names
  df = pd.DataFrame(data, columns=columns)

  conn.close()
  return df
