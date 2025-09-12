import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
from PIL import Image
import io
import os
from datetime import datetime, date
import json
import traceback
import base64

# Configure the page
st.set_page_config(
    page_title="Receipt Tracker",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'expenses' not in st.session_state:
    st.session_state.expenses = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    # Check if we need to migrate the existing table
    cursor.execute("PRAGMA table_info(expenses)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if not columns:
        # Create new table with updated schema
        cursor.execute('''
            CREATE TABLE expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant TEXT,
                date DATE,
                total REAL,
                currency TEXT DEFAULT 'USD',
                category TEXT,
                description TEXT,
                receipt_image BLOB,
                receipt_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    else:
        # Add new columns if they don't exist (for existing databases)
        if 'currency' not in columns:
            cursor.execute('ALTER TABLE expenses ADD COLUMN currency TEXT DEFAULT "USD"')
        if 'description' not in columns:
            cursor.execute('ALTER TABLE expenses ADD COLUMN description TEXT')
        if 'receipt_image' not in columns:
            cursor.execute('ALTER TABLE expenses ADD COLUMN receipt_image BLOB')
        # Rename items to description if items exists
        if 'items' in columns and 'description' not in columns:
            cursor.execute('ALTER TABLE expenses RENAME COLUMN items TO description')
    
    conn.commit()
    conn.close()

def get_gemini_api_key():
    """Get Gemini API key from environment or Streamlit secrets"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        try:
            api_key = st.secrets['GEMINI_API_KEY']
        except:
            return None
    return api_key

def configure_gemini():
    """Configure Gemini API"""
    api_key = get_gemini_api_key()
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def process_receipt_with_gemini(image_bytes):
    """Process receipt image with Gemini 2.0 Flash API"""
    try:
        if not configure_gemini():
            return None, "Gemini API key not found. Please set GEMINI_API_KEY environment variable or add it to Streamlit secrets."
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create prompt for receipt extraction
        prompt = """
        Analyze this receipt image and extract the following information in JSON format:
        {
            "merchant": "store/restaurant name",
            "date": "YYYY-MM-DD format",
            "total": "numerical value only (e.g., 25.99)",
            "currency": "three-letter currency code (e.g., USD, GBP, EUR, DKK)",
            "category": "one of: Food & Dining, Transportation, Shopping, Entertainment, Healthcare, Utilities, Other",
            "description": "descriptive summary of the expense"
        }
        
        Instructions:
        - For merchant, extract the business name clearly shown on the receipt
        - For date, convert to YYYY-MM-DD format
        - For total, extract the final amount paid (not subtotal)
        - For currency, identify the currency from symbols (£=GBP, €=EUR, $=USD, kr=DKK, etc.) or text
        - For category, choose the most appropriate category based on the merchant/items
        - For description, create a natural summary like:
          * "Evening meal at [restaurant name]" for restaurants
          * "Stay at [hotel name] for [X] nights" for hotels
          * "Grocery shopping at [store name]" for supermarkets
          * "Fuel purchase at [station name]" for gas stations
          * "Flight ticket from [origin] to [destination]" for airlines
          * "Taxi ride in [city]" for transportation
        - Return only valid JSON, no additional text
        """
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        # Parse JSON response
        try:
            # Clean response text
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            data = json.loads(response_text)
            
            # Validate and clean data
            cleaned_data = {
                'merchant': str(data.get('merchant', '')).strip(),
                'date': str(data.get('date', str(date.today()))),
                'total': float(data.get('total', 0.0)),
                'currency': str(data.get('currency', 'USD')).strip().upper(),
                'category': str(data.get('category', 'Other')),
                'description': str(data.get('description', '')).strip()
            }
            
            return cleaned_data, None
            
        except json.JSONDecodeError:
            return None, f"Failed to parse response as JSON: {response.text}"
            
    except Exception as e:
        return None, f"Error processing receipt: {str(e)}"

def save_expense_to_db(expense_data, filename, image_data=None):
    """Save expense to SQLite database"""
    try:
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO expenses (merchant, date, total, currency, category, description, receipt_image, receipt_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            expense_data['merchant'],
            expense_data['date'],
            expense_data['total'],
            expense_data['currency'],
            expense_data['category'],
            expense_data['description'],
            image_data,
            filename
        ))
        
        conn.commit()
        conn.close()
        return True, "Expense saved successfully!"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def load_expenses_from_db():
    """Load all expenses from database"""
    try:
        conn = sqlite3.connect('expenses.db')
        df = pd.read_sql_query('''
            SELECT id, merchant, date, total, currency, category, description, receipt_filename, created_at
            FROM expenses
            ORDER BY date DESC, created_at DESC
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading expenses: {str(e)}")
        return pd.DataFrame()

def delete_expense(expense_id):
    """Delete expense from database"""
    try:
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting expense: {str(e)}")
        return False

# Initialize database
init_database()

# App Title
st.title("🧾 Receipt Expense Tracker")
st.markdown("Upload receipt images and automatically extract expense data using AI")

# Sidebar
with st.sidebar:
    st.header("📊 Expense Summary")
    
    # Load expenses for summary
    expenses_df = load_expenses_from_db()
    
    if not expenses_df.empty:
        # Total expenses
        total_amount = expenses_df['total'].sum()
        st.metric("Total Expenses", f"${total_amount:.2f}")
        
        # Expenses by category
        st.subheader("By Category")
        category_summary = expenses_df.groupby('category')['total'].sum().sort_values(ascending=False)
        for category, amount in category_summary.items():
            st.write(f"**{category}:** ${amount:.2f}")
        
        # Filters
        st.subheader("🔍 Filters")
        
        # Category filter
        categories = ['All'] + list(expenses_df['category'].unique())
        selected_category = st.selectbox("Category", categories)
        
        # Date range filter
        if not expenses_df.empty:
            min_date = pd.to_datetime(expenses_df['date']).min().date()
            max_date = pd.to_datetime(expenses_df['date']).max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
    else:
        st.info("No expenses recorded yet")
        selected_category = 'All'
        date_range = None

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Receipt")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a receipt image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of your receipt"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Receipt", use_container_width=True)
        
        # Process button
        if st.button("🤖 Extract Data", type="primary"):
            with st.spinner("Processing receipt with AI..."):
                # Get image bytes
                image_bytes = uploaded_file.getvalue()
                
                # Process with Gemini
                extracted_data, error = process_receipt_with_gemini(image_bytes)
                
                if error:
                    st.error(error)
                    if "API key" in error:
                        st.info("💡 To use AI extraction, set your Gemini API key in environment variables or Streamlit secrets")
                else:
                    st.session_state.extracted_data = extracted_data
                    st.session_state.processing_complete = True
                    st.success("✅ Data extracted successfully!")

with col2:
    st.header("✏️ Expense Details")
    
    # Form for editing/entering expense data
    with st.form("expense_form", clear_on_submit=True):
        # Pre-populate with extracted data if available
        default_data = st.session_state.extracted_data if st.session_state.processing_complete else {}
        
        merchant = st.text_input(
            "Merchant",
            value=default_data.get('merchant', ''),
            placeholder="Store or restaurant name"
        )
        
        expense_date = st.date_input(
            "Date",
            value=datetime.strptime(default_data.get('date', str(date.today())), '%Y-%m-%d').date() 
                  if default_data.get('date') else date.today()
        )
        
        total = st.number_input(
            "Total Amount",
            min_value=0.0,
            value=float(default_data.get('total', 0.0)),
            step=0.01,
            format="%.2f"
        )
        
        currency = st.selectbox(
            "Currency",
            ['USD', 'GBP', 'EUR', 'DKK', 'SEK', 'NOK', 'CAD', 'AUD', 'JPY', 'CHF'],
            index=0 if not default_data.get('currency') else (
                ['USD', 'GBP', 'EUR', 'DKK', 'SEK', 'NOK', 'CAD', 'AUD', 'JPY', 'CHF'].index(default_data.get('currency'))
                if default_data.get('currency') in ['USD', 'GBP', 'EUR', 'DKK', 'SEK', 'NOK', 'CAD', 'AUD', 'JPY', 'CHF']
                else 0
            )
        )
        
        categories = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Healthcare', 'Utilities', 'Other']
        default_category = default_data.get('category', 'Other')
        category_index = categories.index(default_category) if default_category in categories else 6
        
        category = st.selectbox(
            "Category",
            categories,
            index=category_index
        )
        
        description = st.text_area(
            "Description",
            value=default_data.get('description', ''),
            placeholder="Descriptive summary of the expense (e.g., 'Evening meal at Restaurant Name')"
        )
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Expense", type="primary")
        
        if submitted:
            if merchant and total > 0:
                expense_data = {
                    'merchant': merchant,
                    'date': str(expense_date),
                    'total': total,
                    'currency': currency,
                    'category': category,
                    'description': description
                }
                
                # Convert image to bytes if uploaded
                image_data = None
                if uploaded_file:
                    image_data = uploaded_file.getvalue()
                
                # Save to database
                success, message = save_expense_to_db(
                    expense_data, 
                    uploaded_file.name if uploaded_file else "manual_entry",
                    image_data
                )
                
                if success:
                    st.success(message)
                    # Reset processing state
                    st.session_state.processing_complete = False
                    st.session_state.extracted_data = {}
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in merchant name and amount")

# Expense table
st.header("📋 Expense History")

# Reload expenses
expenses_df = load_expenses_from_db()

if not expenses_df.empty:
    # Apply filters
    filtered_df = expenses_df.copy()
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
            (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
        ]
    
    if not filtered_df.empty:
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filtered Total", f"${filtered_df['total'].sum():.2f}")
        with col2:
            st.metric("Number of Expenses", len(filtered_df))
        with col3:
            st.metric("Average Amount", f"${filtered_df['total'].mean():.2f}")
        
        # Display table
        display_df = filtered_df[['merchant', 'date', 'total', 'currency', 'category', 'description']].copy()
        # Format the total with currency
        display_df['amount'] = display_df.apply(
            lambda row: f"{row['total']:.2f} {row['currency']}", axis=1
        )
        # Select columns for display
        display_df = display_df[['merchant', 'date', 'amount', 'category', 'description']]
        
        # Display dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Delete functionality with selectbox
        if len(filtered_df) > 0:
            st.subheader("🗑️ Delete Expense")
            expense_options = []
            for idx, row in filtered_df.iterrows():
                expense_options.append(f"{row['merchant']} - {row['date']} - {row['total']:.2f} {row['currency']}")
            
            selected_expense = st.selectbox(
                "Select expense to delete:",
                options=["None"] + expense_options,
                key="delete_expense_selector"
            )
            
            if selected_expense != "None":
                selected_idx = expense_options.index(selected_expense)
                expense_id = filtered_df.iloc[selected_idx]['id']
                
                if st.button("🗑️ Confirm Delete", type="secondary"):
                    if delete_expense(expense_id):
                        st.success("Expense deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete expense")
    else:
        st.info("No expenses found matching the selected filters")
else:
    st.info("No expenses recorded yet. Upload a receipt to get started!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Google Gemini AI*")