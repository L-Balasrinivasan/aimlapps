import sqlite3


# Function to execute SQL queries
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

# Example usage:
query = """
    SELECT 
      s.name,
      s.contact_person,
      s.email,
      s.phone,
      s.address,
      s.city
    FROM suppliers AS s
    LEFT JOIN products AS p ON s.supplier_id = p.supplier_id
    WHERE p.product_id IS NULL
"""
data, description = execute_query(query)
print(data)