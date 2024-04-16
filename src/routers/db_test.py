import sqlite3

def execute_query(sql_query):
    conn = sqlite3.connect(r'C:\Ai-product\ai-backend\src\routers\employee.db')
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        rows = cur.fetchall()
        return rows, cur.description
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    finally:
        cur.close()
        conn.close()

results = execute_query("SELECT name, price FROM products")
print(results)