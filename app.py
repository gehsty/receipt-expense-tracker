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
if 'field_config' not in st.session_state:
    st.session_state.field_config = {
        'default_fields': [
            {'name': 'merchant', 'type': 'text', 'required': True, 'active': True},
            {'name': 'date', 'type': 'date', 'required': True, 'active': True},
            {'name': 'total', 'type': 'number', 'required': True, 'active': True},
            {'name': 'currency', 'type': 'dropdown', 'required': False, 'active': True},
            {'name': 'category', 'type': 'dropdown', 'required': False, 'active': True},
            {'name': 'description', 'type': 'textarea', 'required': False, 'active': True}
        ],
        'custom_fields': []
    }

def get_active_fields():
    """Get list of all active field names"""
    active = []
    for field in st.session_state.field_config['default_fields']:
        if field['active']:
            active.append(field['name'])
    active.extend(st.session_state.field_config['custom_fields'])
    return active

def toggle_default_field(field_name):
    """Toggle a default field on/off"""
    for field in st.session_state.field_config['default_fields']:
        if field['name'] == field_name and not field['required']:
            field['active'] = not field['active']
            break

def add_custom_field(field_name):
    """Add a new custom field"""
    if field_name and field_name not in st.session_state.field_config['custom_fields']:
        # Check if it conflicts with default field names
        default_names = [f['name'] for f in st.session_state.field_config['default_fields']]
        if field_name.lower() not in default_names:
            st.session_state.field_config['custom_fields'].append(field_name)
            return True
    return False

def remove_custom_field(field_name):
    """Remove a custom field"""
    if field_name in st.session_state.field_config['custom_fields']:
        st.session_state.field_config['custom_fields'].remove(field_name)

def generate_dynamic_prompt():
    """Generate LLM prompt based on active fields"""
    base_prompt = """
        Analyze this receipt image and extract the following information in JSON format:
        {"""

    # Add default fields that are active
    field_descriptions = {}
    for field in st.session_state.field_config['default_fields']:
        if field['active']:
            if field['name'] == 'merchant':
                field_descriptions['merchant'] = "store/restaurant name"
            elif field['name'] == 'date':
                field_descriptions['date'] = "YYYY-MM-DD format"
            elif field['name'] == 'total':
                field_descriptions['total'] = "numerical value only (e.g., 25.99)"
            elif field['name'] == 'currency':
                field_descriptions['currency'] = "three-letter currency code (e.g., USD, GBP, EUR, DKK)"
            elif field['name'] == 'category':
                field_descriptions['category'] = "one of: Food & Dining, Transportation, Shopping, Entertainment, Healthcare, Utilities, Other"
            elif field['name'] == 'description':
                field_descriptions['description'] = "descriptive summary of the expense"

    # Add custom fields
    for field_name in st.session_state.field_config['custom_fields']:
        field_descriptions[field_name] = f"extract {field_name} if visible on receipt, otherwise leave empty"

    # Build JSON structure
    json_fields = []
    for field_name, description in field_descriptions.items():
        json_fields.append(f'            "{field_name}": "{description}"')

    base_prompt += "\n" + ",\n".join(json_fields) + """
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
        - For custom fields, extract the information if clearly visible on the receipt
        - Return only valid JSON, no additional text
        """

    return base_prompt

def render_expense_form():
    """Render the dynamic expense form based on field configuration"""
    # Pre-populate with extracted data if available
    default_data = st.session_state.get('extracted_data', {}) if st.session_state.get('processing_complete', False) else {}

    form_data = {}

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .field-container {
        margin-bottom: 1rem;
    }
    .remove-btn {
        margin-top: 1.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton > button {
        height: 2.5rem !important;
        padding: 0 0.75rem !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 0.375rem !important;
        border: 1px solid #d1d5db !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #f3f4f6 !important;
        color: #6b7280 !important;
        border: 1px solid #d1d5db !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #e5e7eb !important;
        color: #374151 !important;
    }
    .add-field-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }
    .stContainer > div {
        gap: 0.5rem;
    }
    /* Ensure consistent input heights */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input,
    .stNumberInput > div > div > input {
        height: 2.5rem !important;
    }
    /* Better spacing for form elements */
    .element-container {
        margin-bottom: 1rem !important;
    }
    /* Align remove buttons properly */
    div[data-testid="column"]:nth-child(2) {
        display: flex;
        align-items: flex-end;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render default fields
    for field_config in st.session_state.field_config['default_fields']:
        if field_config['active']:
            field_name = field_config['name']
            field_type = field_config['type']
            is_required = field_config['required']

            label = field_name.replace('_', ' ').title()
            if is_required:
                label += " *"

            # Create container for better alignment
            if not is_required:
                col1, col2 = st.columns([9, 1])
                with col1:
                    form_data[field_name] = render_field_input(field_name, field_type, label, default_data)
                with col2:
                    st.write("")  # Add space to align with input field
                    if st.button("×", key=f"remove_default_{field_name}", help=f"Remove {label}", type="secondary"):
                        toggle_default_field(field_name)
                        st.rerun()
            else:
                form_data[field_name] = render_field_input(field_name, field_type, label, default_data)

    # Render custom fields
    for field_name in st.session_state.field_config['custom_fields']:
        col1, col2 = st.columns([9, 1])
        with col1:
            form_data[field_name] = st.text_input(
                field_name,
                value=default_data.get(field_name, ''),
                key=f"custom_{field_name}_input"
            )
        with col2:
            st.write("")  # Add space to align with input field
            if st.button("×", key=f"remove_custom_{field_name}", help=f"Remove {field_name}", type="secondary"):
                remove_custom_field(field_name)
                st.rerun()

    # Add new field section
    st.markdown('<div class="add-field-section">', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        new_field_name = st.text_input("Add Field:", placeholder="e.g., City of Purchase", key="new_field_input")
    with col2:
        st.write("")  # Add space to align with input field
        if st.button("+ Add", disabled=not new_field_name or new_field_name.strip() == "", type="primary"):
            if add_custom_field(new_field_name.strip()):
                st.rerun()
            else:
                st.error("Field already exists or invalid name")

    st.markdown('</div>', unsafe_allow_html=True)

    return form_data

def render_field_input(field_name, field_type, label, default_data):
    """Render appropriate input widget based on field type"""
    if field_type == 'text':
        return st.text_input(
            label,
            value=default_data.get(field_name, ''),
            placeholder=f"Enter {field_name.replace('_', ' ')}",
            key=f"{field_name}_input"
        )
    elif field_type == 'date':
        default_date = date.today()
        if default_data.get(field_name):
            try:
                default_date = datetime.strptime(default_data.get(field_name), '%Y-%m-%d').date()
            except:
                default_date = date.today()
        return st.date_input(
            label,
            value=default_date,
            key=f"{field_name}_input"
        )
    elif field_type == 'number':
        return st.number_input(
            label,
            min_value=0.0,
            value=float(default_data.get(field_name, 0.0)),
            step=0.01,
            format="%.2f",
            key=f"{field_name}_input"
        )
    elif field_type == 'dropdown':
        if field_name == 'currency':
            options = ['USD', 'GBP', 'EUR', 'DKK', 'SEK', 'NOK', 'CAD', 'AUD', 'JPY', 'CHF']
            default_index = 0
            if default_data.get(field_name) and default_data.get(field_name) in options:
                default_index = options.index(default_data.get(field_name))
        elif field_name == 'category':
            options = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Healthcare', 'Utilities', 'Other']
            default_index = 6  # Default to "Other"
            if default_data.get(field_name) and default_data.get(field_name) in options:
                default_index = options.index(default_data.get(field_name))
        else:
            options = ['Option 1', 'Option 2', 'Option 3']
            default_index = 0

        return st.selectbox(
            label,
            options,
            index=default_index,
            key=f"{field_name}_input"
        )
    elif field_type == 'textarea':
        return st.text_area(
            label,
            value=default_data.get(field_name, ''),
            placeholder=f"Enter {field_name.replace('_', ' ')}",
            key=f"{field_name}_input"
        )
    else:
        # Default to text input
        return st.text_input(
            label,
            value=default_data.get(field_name, ''),
            key=f"{field_name}_input"
        )

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
                custom_fields JSON,
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
        if 'custom_fields' not in columns:
            cursor.execute('ALTER TABLE expenses ADD COLUMN custom_fields JSON')
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
        
        # Create dynamic prompt based on active fields
        prompt = generate_dynamic_prompt()
        
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
            
            # Validate and clean data dynamically based on active fields
            cleaned_data = {}
            active_fields = get_active_fields()

            for field_name in active_fields:
                if field_name in data:
                    if field_name == 'total':
                        try:
                            cleaned_data[field_name] = float(data.get(field_name, 0.0))
                        except:
                            cleaned_data[field_name] = 0.0
                    elif field_name == 'currency':
                        cleaned_data[field_name] = str(data.get(field_name, 'USD')).strip().upper()
                    elif field_name == 'date':
                        cleaned_data[field_name] = str(data.get(field_name, str(date.today())))
                    else:
                        cleaned_data[field_name] = str(data.get(field_name, '')).strip()
                else:
                    # Set default values for missing fields
                    if field_name == 'total':
                        cleaned_data[field_name] = 0.0
                    elif field_name == 'currency':
                        cleaned_data[field_name] = 'USD'
                    elif field_name == 'date':
                        cleaned_data[field_name] = str(date.today())
                    elif field_name == 'category':
                        cleaned_data[field_name] = 'Other'
                    else:
                        cleaned_data[field_name] = ''
            
            return cleaned_data, None
            
        except json.JSONDecodeError:
            return None, f"Failed to parse response as JSON: {response.text}"
            
    except Exception as e:
        return None, f"Error processing receipt: {str(e)}"

def save_expense_to_db(expense_data, custom_fields, filename, image_data=None):
    """Save expense to SQLite database with support for custom fields"""
    try:
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()

        # Convert custom fields to JSON
        custom_fields_json = json.dumps(custom_fields) if custom_fields else None

        cursor.execute('''
            INSERT INTO expenses (merchant, date, total, currency, category, description, custom_fields, receipt_image, receipt_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            expense_data.get('merchant', ''),
            expense_data.get('date', str(date.today())),
            expense_data.get('total', 0.0),
            expense_data.get('currency', 'USD'),
            expense_data.get('category', 'Other'),
            expense_data.get('description', ''),
            custom_fields_json,
            image_data,
            filename
        ))

        conn.commit()
        conn.close()
        return True, "Expense saved successfully!"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def load_expenses_from_db():
    """Load all expenses from database with custom fields support"""
    try:
        conn = sqlite3.connect('expenses.db')
        df = pd.read_sql_query('''
            SELECT id, merchant, date, total, currency, category, description, custom_fields, receipt_filename, created_at
            FROM expenses
            ORDER BY date DESC, created_at DESC
        ''', conn)
        conn.close()

        # Parse custom fields and expand into columns
        if not df.empty and 'custom_fields' in df.columns:
            # Get all unique custom field names across all records
            all_custom_fields = set()
            for idx, row in df.iterrows():
                if row['custom_fields']:
                    try:
                        custom_data = json.loads(row['custom_fields'])
                        all_custom_fields.update(custom_data.keys())
                    except json.JSONDecodeError:
                        pass

            # Add columns for each custom field
            for field_name in all_custom_fields:
                df[field_name] = ''

            # Populate custom field columns
            for idx, row in df.iterrows():
                if row['custom_fields']:
                    try:
                        custom_data = json.loads(row['custom_fields'])
                        for field_name, value in custom_data.items():
                            df.at[idx, field_name] = value
                    except json.JSONDecodeError:
                        pass

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
    
    # Dynamic form based on field configuration
    try:
        # Render dynamic form
        form_data = render_expense_form()

        # Submit button with spacing
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Expense", type="primary", key="save_button", use_container_width=True):
            # Validate required fields
            required_fields = [f['name'] for f in st.session_state.field_config['default_fields'] if f['required'] and f['active']]
            missing_fields = []

            for field in required_fields:
                if not form_data.get(field) or form_data.get(field) == "" or (field == 'total' and form_data.get(field) <= 0):
                    missing_fields.append(field)

            if missing_fields:
                st.error(f"Please fill in required fields: {', '.join(missing_fields)}")
            else:
                # Prepare expense data
                expense_data = {}
                custom_fields = {}

                # Separate default and custom fields
                default_field_names = [f['name'] for f in st.session_state.field_config['default_fields']]

                for field_name, value in form_data.items():
                    if field_name in default_field_names:
                        expense_data[field_name] = value
                    else:
                        custom_fields[field_name] = value

                # Convert image to bytes if uploaded
                image_data = None
                if uploaded_file:
                    image_data = uploaded_file.getvalue()

                # Save to database
                success, message = save_expense_to_db(
                    expense_data,
                    custom_fields,
                    uploaded_file.name if uploaded_file else "manual_entry",
                    image_data
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    except Exception as e:
        st.error(f"Form error: {str(e)}")
        st.info("Try refreshing the page if the error persists.")

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
        
        # Display table with dynamic columns
        base_columns = ['merchant', 'date', 'total', 'currency', 'category', 'description']

        # Get all columns that exist in the dataframe (including custom fields)
        available_columns = [col for col in base_columns if col in filtered_df.columns]
        custom_columns = [col for col in filtered_df.columns if col not in base_columns and col not in ['id', 'custom_fields', 'receipt_filename', 'created_at']]

        # Create display dataframe
        all_display_columns = available_columns + custom_columns
        display_df = filtered_df[all_display_columns].copy()

        # Format the total with currency if both exist
        if 'total' in display_df.columns and 'currency' in display_df.columns:
            display_df['amount'] = display_df.apply(
                lambda row: f"{row['total']:.2f} {row['currency']}", axis=1
            )
            # Remove separate total and currency columns, keep amount
            columns_to_show = [col for col in all_display_columns if col not in ['total', 'currency']]
            columns_to_show.insert(2, 'amount')  # Insert amount after date
            display_df = display_df[columns_to_show]
        
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