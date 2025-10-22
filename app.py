# ==============================
# 🧠 Text-to-SQL Streamlit App (PostgreSQL + Phi-3-mini)
# ==============================

import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------------------------
# 🚀 GPU Setup
# ------------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# ------------------------------
# 🧠 Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device)
    return pipe

pipe = load_model()

# ------------------------------
# 🗄️ PostgreSQL Connection
# ------------------------------
st.sidebar.header("Database Connection")
host = st.sidebar.text_input("Host", "localhost")
port = st.sidebar.text_input("Port", "5432")
database = st.sidebar.text_input("Database", "olist")
user = st.sidebar.text_input("User", "postgres")
password = st.sidebar.text_input("Password", type="password")

# Create engine
engine = None
if st.sidebar.button("🔗 Connect to Database"):
    try:
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
        with engine.connect() as conn:
            st.sidebar.success("✅ Connected successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {e}")

# ------------------------------
# 📜 Schema Context
# ------------------------------
SCHEMA_HINT = """
Tables and Columns:
- customers(customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
- orders(order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
- order_items(order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
- products(product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
- sellers(seller_id, seller_zip_code_prefix, seller_city, seller_state)
"""

# ------------------------------
# 🧩 English → SQL Conversion
# ------------------------------
def english_to_sql(prompt):
    system_prompt = f"""
You are an expert PostgreSQL data analyst.
You have access to this schema:
{SCHEMA_HINT}

Convert the following English instruction into a valid PostgreSQL SQL query.
- Output only the SQL query.
- Do not include explanations or markdown.
- Ensure all SELECT columns are either aggregated or in GROUP BY.

Instruction: {prompt}
SQL:
"""
    response = pipe(system_prompt, max_new_tokens=250, temperature=0.3, do_sample=False)
    text = response[0]["generated_text"]
    sql = text.split("SQL:")[-1].strip()
    sql = sql.replace("```sql", "").replace("```", "").split("Instruction:")[0].strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql

# ------------------------------
# 🧾 Execute SQL Query
# ------------------------------
def run_sql_query(query):
    if not engine:
        st.error("⚠️ Please connect to your PostgreSQL database first.")
        return None
    try:
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ SQL Execution Error: {e}")
        return None

# ------------------------------
# 🎨 Visualization Helper
# ------------------------------
def visualize(df):
    if df is not None and not df.empty:
        st.dataframe(df)
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            st.bar_chart(df.set_index(df.columns[0])[num_cols[0]])
    else:
        st.warning("⚠️ No data to visualize.")

# ------------------------------
# 🧠 Streamlit UI
# ------------------------------
st.title("🧠 Text-to-SQL with Phi-3-mini + PostgreSQL")

prompt = st.text_area("Enter your question in English:", 
    placeholder="e.g. Show me top 10 customers by total spending")

if st.button("Generate SQL"):
    sql_query = english_to_sql(prompt)
    st.code(sql_query, language="sql")

    df = run_sql_query(sql_query)
    visualize(df)
