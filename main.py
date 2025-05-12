# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, g
# import sqlite3 # sqlite3 আর প্রয়োজন নেই
import psycopg2 # psycopg2 যোগ করা হলো
import psycopg2.extras # DictCursor এর জন্য
from database import get_db_connection, init_db, DATABASE_URL # DATABASE_URL এখানেও ইম্পোর্ট করা হলো
import os
from datetime import datetime
import pytz # সময় অঞ্চলের জন্য

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_please_change_it_for_production'

# PostgreSQL DATABASE_URL এখন database.py থেকে আসছে
# TINYMCE_API_KEY আগের মতোই থাকবে

# --- বাংলা সংখ্যায় রূপান্তর ফাংশন --- (অপরিবর্তিত)
@app.template_filter('to_bengali_numerals')
def to_bengali_numerals(number_input):
    number_str = str(number_input)
    english_to_bengali = {
        '0': '০', '1': '১', '2': '২', '3': '৩', '4': '৪',
        '5': '৫', '6': '৬', '7': '৭', '8': '৮', '9': '৯'
    }
    return "".join([english_to_bengali.get(digit, digit) for digit in number_str])

# --- BST datetime প্রস্তুত করার হেল্পার ফাংশন --- (অপরিবর্তিত)
def _prepare_bst_datetime(dt_obj_input):
    if not dt_obj_input:
        return None
    dt_obj = None
    if isinstance(dt_obj_input, str):
        processed_dt_str = dt_obj_input.replace(" ", "T")
        try:
            dt_obj = datetime.fromisoformat(processed_dt_str)
        except ValueError:
            common_formats = [
                '%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z', # ISO with TZ
                '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z', # Space separator with TZ
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'
            ]
            parsed_success = False
            for fmt in common_formats:
                try:
                    # psycopg2 থেকে আসা টাইমস্ট্যাম্পগুলোতে টাইমজোন তথ্য থাকতে পারে
                    if '%z' in fmt: # যদি ফরম্যাটে টাইমজোন অফসেট থাকে
                        dt_obj = datetime.strptime(dt_obj_input, fmt)
                    else: # টাইমজোন ছাড়া ফরম্যাট
                        # যদি ইনপুট স্ট্রিং-এ টাইমজোন না থাকে, তবে সাধারণ পার্সিং
                        temp_dt_obj = datetime.strptime(dt_obj_input.split('+')[0].split('.')[0], fmt.split('.')[0])
                        # psycopg2 অনেক সময় UTC তে দেয়, যদি না অ্যাপ্লিকেশন লেভেলে টাইমজোন সেট করা হয়
                        # এখানে আমরা ধরে নিচ্ছি যে ডাটাবেস থেকে আসা স্ট্রিং হয় naive অথবা UTC
                        # যদি naive হয়, তবে pytz.utc দিয়ে localize করা হবে নিচে
                        dt_obj = temp_dt_obj
                    parsed_success = True
                    break
                except ValueError:
                    continue
            if not parsed_success:
                return f"ত্রুটিপূর্ণ তারিখ স্ট্রিং ({dt_obj_input})"
    elif isinstance(dt_obj_input, datetime):
        dt_obj = dt_obj_input
    else:
        return f"অপরিচিত তারিখ টাইপ ({type(dt_obj_input)})"

    if not isinstance(dt_obj, datetime):
        return dt_obj # Return as is if it's already a processed error string

    utc_tz = pytz.utc
    if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
        utc_dt = utc_tz.localize(dt_obj) # Naive datetime কে UTC তে localize করা
    else: # যদি টাইমজোন তথ্য থাকে, UTC তে রূপান্তর করুন
        utc_dt = dt_obj.astimezone(utc_tz)

    bst_tz = pytz.timezone('Asia/Dhaka')
    return utc_dt.astimezone(bst_tz)


# --- কাস্টম Jinja2 ফিল্টার (তারিখ ও সময়) --- (অপরিবর্তিত)
@app.template_filter('format_bangla_datetime')
def format_bangla_datetime_filter(dt_obj_input):
    bst_dt = _prepare_bst_datetime(dt_obj_input)
    if not bst_dt or isinstance(bst_dt, str): # Check if it's an error string from _prepare_bst_datetime
        return {'date_str': bst_dt if isinstance(bst_dt, str) else 'N/A', 'time_str': ''}

    bangla_months = ['জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন', 'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর']
    day_bn = to_bengali_numerals(bst_dt.day)
    day_str = f"{day_bn}ই" # Added 'ই' suffix for day

    month_name_bn = bangla_months[bst_dt.month - 1]
    year_bn = to_bengali_numerals(bst_dt.year)
    year_str = f"{year_bn} খ্রিস্টাব্দ"
    date_part_str = f"{day_str} {month_name_bn} {year_str}"

    hour_24 = bst_dt.hour
    minute = bst_dt.minute
    time_period = ""
    hour_12_western = int(bst_dt.strftime('%I')) # Get 12-hour format
    hour_12_bn = to_bengali_numerals(hour_12_western)
    minute_bn = to_bengali_numerals(f"{minute:02d}") # Ensure two digits for minute

    if 6 <= hour_24 < 12: time_period = "সকাল"
    elif 12 <= hour_24 < 15: time_period = "দুপুর" # Adjusted to 12-3 PM for দুপুর
    elif 15 <= hour_24 < 18: time_period = "বিকাল" # Adjusted to 3-6 PM for বিকাল
    elif 18 <= hour_24 < 20: time_period = "সন্ধ্যা" # Adjusted to 6-8 PM for সন্ধ্যা
    else: # Covers 8 PM to 5:59 AM
        time_period = "রাত"

    time_part_str = f"{time_period} {hour_12_bn}:{minute_bn} মিনিট"
    return {'date_str': date_part_str, 'time_str': time_part_str}

# --- কাস্টম Jinja2 ফিল্টার (শুধুমাত্র তারিখ) --- (অপরিবর্তিত)
@app.template_filter('format_bangla_date')
def format_bangla_date_filter(dt_obj_input):
    bst_dt = _prepare_bst_datetime(dt_obj_input)
    if not bst_dt or isinstance(bst_dt, str): # Check if it's an error string
        return bst_dt if isinstance(bst_dt, str) else 'N/A'

    bangla_months = ['জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন', 'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর']
    day_bn = to_bengali_numerals(bst_dt.day)
    day_str = f"{day_bn}ই" # Added 'ই' suffix
    month_name_bn = bangla_months[bst_dt.month - 1]
    year_bn = to_bengali_numerals(bst_dt.year)
    year_str = f"{year_bn} খ্রিস্টাব্দ"
    return f"{day_str} {month_name_bn} {year_str}"

# গ্লোবাল ভেরিয়েবল ইনজেক্ট করার জন্য (অপরিবর্তিত)
@app.context_processor
def inject_global_vars():
    dhaka_tz = pytz.timezone('Asia/Dhaka')
    current_time_dhaka = datetime.now(dhaka_tz)
    return {
        'current_year_bn': to_bengali_numerals(current_time_dhaka.year)
    }

# --- ডাটাবেস কানেকশন ---
@app.before_request
def before_request():
    # This function runs before each request.
    # It establishes a database connection and cursor, storing them in Flask's 'g' object.
    # 'g' is a context-local object that can be used to store resources during a request.
    g.db_conn = get_db_connection() # Database connection object
    g.db_cursor = g.db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # Database cursor, returns rows as dictionaries

@app.teardown_request
def teardown_request(exception):
    # This function runs after each request, even if an exception occurred.
    # It ensures that the database cursor and connection are closed properly.
    cursor = getattr(g, 'db_cursor', None) # Get the cursor from 'g', if it exists
    if cursor is not None:
        cursor.close() # Close the cursor
    conn = getattr(g, 'db_conn', None) # Get the connection from 'g', if it exists
    if conn is not None:
        conn.close() # Close the connection

# --- Helper Functions (পরিবর্তিত) ---
def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False, get_last_id=False):
    """
    PostgreSQL এর জন্য জেনেরিক কোয়েরি এক্সিকিউশন ফাংশন।
    Executes a given SQL query with optional parameters.
    Can fetch one row, all rows, commit changes, or return the ID of the last inserted row.
    """
    try:
        if params:
            g.db_cursor.execute(query, params) # Execute query with parameters
        else:
            g.db_cursor.execute(query) # Execute query without parameters

        result = None
        if get_last_id: # For INSERT ... RETURNING id
            result = g.db_cursor.fetchone()[0] # Fetch the first column (id) of the first row
        elif fetchone:
            result = g.db_cursor.fetchone() # Fetch a single row
        elif fetchall:
            result = g.db_cursor.fetchall() # Fetch all rows

        if commit:
            g.db_conn.commit() # Commit the transaction
        return result
    except (Exception, psycopg2.Error) as error:
        g.db_conn.rollback() # Rollback the transaction in case of an error
        print(f"PostgreSQL Error: {error}")
        flash(f"ডাটাবেস অপারেশনে সমস্যা হয়েছে: {error}", "error")
        # More detailed error handling can be added here if needed
        return None # Or return something specific based on the error


def get_story(story_id):
    # Fetches a single story by its ID.
    return execute_query('SELECT * FROM stories WHERE id = %s', (story_id,), fetchone=True)

def get_story_parts(story_id):
    # Fetches all parts of a story, ordered by part_order and then by ID.
    return execute_query('SELECT * FROM story_parts WHERE story_id = %s ORDER BY part_order ASC, id ASC', (story_id,), fetchall=True)

def get_pages_for_owner(owner_type, owner_id):
    # Fetches all content pages for a given owner (story or story_part), ordered by page_number.
    return execute_query('SELECT * FROM content_pages WHERE owner_type = %s AND owner_id = %s ORDER BY page_number ASC', (owner_type, owner_id), fetchall=True)

def get_content_page(page_id):
    # Fetches a single content page by its ID.
    return execute_query('SELECT * FROM content_pages WHERE id = %s', (page_id,), fetchone=True)

def get_next_page_number(owner_type, owner_id):
    # Calculates the next available page number for a given owner.
    last_page = execute_query('SELECT MAX(page_number) as max_page FROM content_pages WHERE owner_type = %s AND owner_id = %s', (owner_type, owner_id), fetchone=True)
    return (last_page['max_page'] if last_page and last_page['max_page'] else 0) + 1

# --- Public Routes (অপরিবর্তিত, helper ফাংশন ব্যবহারের কারণে) ---
@app.route('/')
def index():
    # Main page, displays all stories.
    stories = execute_query('SELECT * FROM stories ORDER BY created_at DESC', fetchall=True)
    return render_template('index.html', stories=stories, search_query=request.args.get('q', ''))

@app.route('/story/<int:story_id>')
@app.route('/story/<int:story_id>/page/<int:page_num>')
def story_detail(story_id, page_num=1):
    # Displays details of a story, including its parts or pages.
    story = get_story(story_id)
    if not story:
        flash('গল্প/নোট খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('index'))

    if story['has_parts']:
        parts = get_story_parts(story_id)
        return render_template('story_detail.html', story=story, parts=parts, current_page_num=None, total_pages=None, content_page=None, search_query=request.args.get('q', ''))
    else:
        pages = get_pages_for_owner('story', story_id)
        if not pages:
            return render_template('story_detail.html', story=story, parts=None, current_page_num=0, total_pages=0, content_page=None, search_query=request.args.get('q', ''))

        total_pages = len(pages)
        if not (1 <= page_num <= total_pages) and total_pages > 0 : # Added total_pages > 0 condition
            flash('অনুরোধকৃত পৃষ্ঠা নম্বরটি সঠিক নয়।', 'warning')
            page_num = 1

        content_page = next((p for p in pages if p['page_number'] == page_num), None)
        if not content_page and pages: # If specific page not found, default to first
             content_page = pages[0]
             page_num = content_page['page_number']


        return render_template('story_detail.html', story=story, parts=None, current_page_num=page_num, total_pages=total_pages, content_page=content_page, search_query=request.args.get('q', ''))

@app.route('/story/<int:story_id>/part/<int:part_id>')
@app.route('/story/<int:story_id>/part/<int:part_id>/page/<int:page_num>')
def part_detail(story_id, part_id, page_num=1):
    # Displays details of a specific part of a story, including its pages.
    story = get_story(story_id)
    part = execute_query('SELECT * FROM story_parts WHERE id = %s AND story_id = %s', (part_id, story_id), fetchone=True)

    if not story or not part:
        flash('অনুরোধকৃত অংশটি খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('index'))

    pages = get_pages_for_owner('story_part', part_id)
    if not pages:
        return render_template('part_detail.html', story=story, part=part, current_page_num=0, total_pages=0, content_page=None, search_query=request.args.get('q', ''))

    total_pages = len(pages)
    if not (1 <= page_num <= total_pages) and total_pages > 0: # Added total_pages > 0 condition
        flash('অনুরোধকৃত পৃষ্ঠা নম্বরটি সঠিক নয়।', 'warning')
        page_num = 1

    content_page = next((p for p in pages if p['page_number'] == page_num), None)
    if not content_page and pages: # If specific page not found, default to first
        content_page = pages[0]
        page_num = content_page['page_number']


    return render_template('part_detail.html', story=story, part=part, current_page_num=page_num, total_pages=total_pages, content_page=content_page, search_query=request.args.get('q', ''))

@app.route('/search')
def search_results_page():
    # Displays search results for stories.
    query = request.args.get('q', '').strip()
    if not query:
        flash('অনুসন্ধানের জন্য অনুগ্রহ করে কিছু লিখুন।', 'warning')
        return redirect(url_for('index'))

    search_term = f"%{query}%"
    # Using ILIKE for case-insensitive search in PostgreSQL
    stories_results = execute_query(
        'SELECT * FROM stories WHERE title ILIKE %s OR summary ILIKE %s ORDER BY created_at DESC',
        (search_term, search_term), fetchall=True
    )
    page_title = f"'{query}' এর জন্য অনুসন্ধানের ফলাফল"
    return render_template('search_results.html', stories=stories_results, search_query=query, page_title=page_title)

# --- Admin Routes ---
@app.route('/admin')
def admin_dashboard():
    # Admin dashboard, shows a list of stories with part/page counts.
    stories_data = execute_query(
        '''SELECT s.*,
                (SELECT COUNT(id) FROM story_parts sp WHERE sp.story_id = s.id) as part_count,
                (SELECT COUNT(id) FROM content_pages cp WHERE cp.owner_type='story' AND cp.owner_id=s.id) as story_page_count
           FROM stories s ORDER BY s.created_at DESC''', fetchall=True
    )
    stories = []
    if stories_data: # Check if stories_data is not None
        for row_data in stories_data:
            story_dict = dict(row_data) # DictCursor already returns dict-like rows
            if row_data['has_parts']:
                story_dict['display_count'] = row_data['part_count']
                story_dict['count_label'] = 'টি অংশ'
            else:
                story_dict['display_count'] = row_data['story_page_count']
                story_dict['count_label'] = 'টি পৃষ্ঠা'
            stories.append(story_dict)
    return render_template('admin/admin_dashboard.html', stories=stories)


@app.route('/admin/story/add', methods=['GET', 'POST'])
def add_story():
    # Route to add a new story or note.
    if request.method == 'POST':
        title = request.form['title']
        summary = request.form.get('summary', '')
        has_parts = 'has_parts' in request.form
        if not title:
            flash('শিরোনাম আবশ্যক।', 'error')
        else:
            new_story_id = execute_query(
                'INSERT INTO stories (title, summary, has_parts) VALUES (%s, %s, %s) RETURNING id',
                (title, summary, has_parts), commit=True, get_last_id=True
            )
            if new_story_id:
                flash('নতুন গল্প/নোট সফলভাবে যোগ করা হয়েছে।', 'success')
                return redirect(url_for('edit_story', story_id=new_story_id))
            # else: flash message is handled by execute_query on error
    return render_template('admin/add_edit_story.html', story=None, pages=None, story_parts_with_pages=None, action_url=url_for('add_story'), page_title="নতুন গল্প/নোট যোগ করুন")


@app.route('/admin/story/edit/<int:story_id>', methods=['GET', 'POST'])
def edit_story(story_id):
    # Route to edit an existing story or note.
    story_row = get_story(story_id)
    if not story_row:
        flash('গল্প খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('admin_dashboard'))
    story = dict(story_row) # Convert row to dict if not already

    story_pages = None
    story_parts_with_pages = []
    if not story['has_parts']:
        story_pages = get_pages_for_owner('story', story_id)
    else:
        parts_raw = get_story_parts(story_id)
        if parts_raw:
            for part_raw in parts_raw:
                part_dict = dict(part_raw)
                part_pages_list = get_pages_for_owner('story_part', part_raw['id'])
                part_dict['pages'] = part_pages_list if part_pages_list else []
                story_parts_with_pages.append(part_dict)

    if request.method == 'POST':
        title = request.form['title']
        summary = request.form.get('summary', '')
        has_parts_form = 'has_parts' in request.form
        if not title:
            flash('শিরোনাম আবশ্যক।', 'error')
        else:
            original_has_parts = story['has_parts']
            can_change_type = True
            # Check if trying to change story type (single/multi-part) when content exists
            if original_has_parts != has_parts_form:
                content_exists = False
                if original_has_parts: # Was multi-part, changing to single
                    if story_parts_with_pages: content_exists = True
                else: # Was single, changing to multi-part
                    if story_pages: content_exists = True

                if content_exists:
                    flash('গল্পের ধরন (একক/বহু-অংশ) পরিবর্তন করার জন্য, অনুগ্রহ করে প্রথমে সকল পৃষ্ঠা/অংশ ডিলিট করুন। বর্তমানে এই পরিবর্তন সম্ভব নয় কারণ কনটেন্ট বিদ্যমান।', 'warning')
                    can_change_type = False

            if can_change_type:
                execute_query(
                    'UPDATE stories SET title = %s, summary = %s, has_parts = %s WHERE id = %s',
                    (title, summary, has_parts_form, story_id), commit=True
                )
                # execute_query returns None on successful commit without fetch
                # We assume success if no exception was raised and handled by execute_query
                flash_message = 'গল্প/নোটের তথ্য'
                if original_has_parts != has_parts_form:
                     flash_message += ' এবং ধরন'
                flash_message += ' সফলভাবে আপডেট করা হয়েছে।'
                flash(flash_message, 'success')
                return redirect(url_for('edit_story', story_id=story_id)) # Refresh on success

    return render_template('admin/add_edit_story.html', story=story, pages=story_pages, story_parts_with_pages=story_parts_with_pages, action_url=url_for('edit_story', story_id=story_id), page_title="গল্প/নোট সম্পাদনা করুন")


@app.route('/admin/story/delete/<int:story_id>', methods=['POST'])
def delete_story(story_id):
    # Route to delete a story and its related content.
    story = get_story(story_id)
    if not story:
        flash('ডিলিট করার জন্য গল্প/নোট খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('admin_dashboard'))

    # Assuming ON DELETE CASCADE is set for story_parts foreign key in PostgreSQL.
    # Content pages related to 'story' type need manual deletion.
    # Content pages related to 'story_part' type also need manual deletion if CASCADE is not on their owner_id.
    # For safety, explicitly delete child content first.

    # Delete content pages directly associated with the story
    execute_query("DELETE FROM content_pages WHERE owner_type = 'story' AND owner_id = %s", (story_id,), commit=False)

    if story['has_parts']:
        parts = get_story_parts(story_id)
        if parts:
            for part in parts:
                # Delete content pages associated with each part
                execute_query("DELETE FROM content_pages WHERE owner_type = 'story_part' AND owner_id = %s", (part['id'],), commit=False)
                # Story parts themselves will be deleted by CASCADE when the story is deleted,
                # or can be deleted explicitly if CASCADE is not reliable or not set on story_parts.
                # execute_query("DELETE FROM story_parts WHERE id = %s", (part['id'],), commit=False) # Optional explicit delete

    # Now delete the story. If ON DELETE CASCADE is set for story_parts.story_id,
    # related story_parts will be deleted automatically.
    execute_query('DELETE FROM stories WHERE id = %s', (story_id,), commit=True) # Final commit

    flash(f"গল্প '{story['title']}' এবং এর সকল সম্পর্কিত ডেটা সফলভাবে ডিলিট করা হয়েছে।", 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/story/<int:story_id>/part/add', methods=['POST'])
def add_part(story_id):
    # Route to add a new part to a story.
    story = get_story(story_id)
    if not story or not story['has_parts']:
        flash('এই গল্পে অংশ যোগ করা যাবে না অথবা গল্পটি পাওয়া যায়নি।', 'error')
        return redirect(url_for('edit_story', story_id=story_id))

    part_title = request.form.get('part_title')
    summary = request.form.get('summary', '')
    part_order_str = request.form.get('part_order', '0')

    if not part_title:
        flash('অংশের শিরোনাম আবশ্যক।', 'error')
    else:
        try:
            part_order = int(part_order_str)
            if part_order < 0: part_order = 0 # Ensure non-negative order
        except ValueError:
            flash('অংশের ক্রম একটি সংখ্যা হতে হবে।', 'error')
            part_order = 0 # Default value

        new_part_id = execute_query(
            'INSERT INTO story_parts (story_id, part_title, summary, part_order) VALUES (%s, %s, %s, %s) RETURNING id',
            (story_id, part_title, summary, part_order), commit=True, get_last_id=True
        )
        if new_part_id:
            flash('নতুন অংশ সফলভাবে যোগ করা হয়েছে। এখন আপনি এই অংশের জন্য পৃষ্ঠা যোগ করতে পারেন।', 'success')
            return redirect(url_for('manage_story_part', part_id=new_part_id))
        # else: flash message handled by execute_query
    return redirect(url_for('edit_story', story_id=story_id))


@app.route('/admin/part/<int:part_id>/manage', methods=['GET', 'POST'])
def manage_story_part(part_id):
    # Route to manage (edit info and list pages of) a story part.
    part_row = execute_query('SELECT * FROM story_parts WHERE id = %s', (part_id,), fetchone=True)
    if not part_row:
        flash('গল্পের অংশ খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('admin_dashboard'))
    part = dict(part_row)
    story = get_story(part['story_id']) # Fetch the parent story for context

    if request.method == 'POST':
        part_title = request.form.get('part_title')
        summary = request.form.get('summary', '')
        part_order_str = request.form.get('part_order', str(part['part_order']))
        if not part_title:
            flash('অংশের শিরোনাম আবশ্যক।', 'error')
        else:
            try:
                part_order = int(part_order_str)
                if part_order < 0: part_order = 0
            except ValueError:
                flash('অংশের ক্রম একটি সংখ্যা হতে হবে।', 'error')
                part_order = part['part_order'] # Keep original if new value is invalid

            execute_query(
                'UPDATE story_parts SET part_title = %s, summary = %s, part_order = %s WHERE id = %s',
                (part_title, summary, part_order, part_id), commit=True
            )
            flash(f"অংশ '{part_title}' এর তথ্য সফলভাবে আপডেট করা হয়েছে।", 'success')
            return redirect(url_for('manage_story_part', part_id=part_id)) # Refresh

    pages = get_pages_for_owner('story_part', part_id)
    return render_template('admin/manage_part_pages.html', story=story, part=part, pages=pages if pages else [], page_title=f"'{part['part_title']}' অংশের পৃষ্ঠা পরিচালনা")

@app.route('/admin/part/delete/<int:part_id>', methods=['POST'])
def delete_part(part_id):
    # Route to delete a story part and its associated pages.
    part = execute_query('SELECT * FROM story_parts WHERE id = %s', (part_id,), fetchone=True)
    story_id_for_redirect = request.form.get('story_id_for_redirect') # Get from form for fallback

    if not part:
        flash('ডিলিট করার জন্য অংশটি খুঁজে পাওয়া যায়নি।', 'error')
        if story_id_for_redirect:
            return redirect(url_for('edit_story', story_id=int(story_id_for_redirect)))
        return redirect(url_for('admin_dashboard'))

    story_id_for_redirect = part['story_id'] # Correct story_id from the part itself

    # First, delete content pages associated with this part
    execute_query("DELETE FROM content_pages WHERE owner_type = 'story_part' AND owner_id = %s", (part_id,), commit=False)
    # Then, delete the story part itself
    execute_query('DELETE FROM story_parts WHERE id = %s', (part_id,), commit=True) # Commit here

    flash(f"অংশ '{part['part_title']}' এবং এর সকল পৃষ্ঠা সফলভাবে ডিলিট করা হয়েছে।", 'success')
    return redirect(url_for('edit_story', story_id=story_id_for_redirect))


@app.route('/admin/owner/<string:owner_type>/<int:owner_id>/page/manage', methods=['GET', 'POST'])
@app.route('/admin/owner/<string:owner_type>/<int:owner_id>/page/manage/<int:page_id>', methods=['GET', 'POST'])
def manage_content_page(owner_type, owner_id, page_id=None):
    # Route to add or edit a content page for a story or a story part.
    if owner_type not in ['story', 'story_part']:
        flash('অপরিচিত মালিকের ধরন।', 'error'); return redirect(url_for('admin_dashboard'))

    owner_entity_row = None
    redirect_url = url_for('admin_dashboard') # Default redirect
    page_owner_title_for_breadcrumbs = ""

    if owner_type == 'story':
        owner_entity_row = get_story(owner_id)
        redirect_url = url_for('edit_story', story_id=owner_id)
        if owner_entity_row: page_owner_title_for_breadcrumbs = owner_entity_row['title']
    else: # owner_type == 'story_part'
        # Fetch part with its story title for breadcrumbs/context
        owner_entity_row = execute_query('SELECT sp.*, s.title as story_title FROM story_parts sp JOIN stories s ON sp.story_id = s.id WHERE sp.id = %s', (owner_id,), fetchone=True)
        redirect_url = url_for('manage_story_part', part_id=owner_id)
        if owner_entity_row: page_owner_title_for_breadcrumbs = f"{owner_entity_row['story_title']} - {owner_entity_row['part_title']}"

    if not owner_entity_row:
        flash(f"{owner_type} ({owner_id}) খুঁজে পাওয়া যায়নি।", 'error'); return redirect(url_for('admin_dashboard'))

    owner_entity = dict(owner_entity_row)
    page_data_row = None
    current_page_title = "" # Initialize current_page_title

    if page_id: # Editing an existing page
        page_data_row = get_content_page(page_id)
        if not page_data_row or page_data_row['owner_type'] != owner_type or page_data_row['owner_id'] != owner_id:
            flash('সম্পাদনার জন্য পৃষ্ঠাটি খুঁজে পাওয়া যায়নি।', 'error'); return redirect(redirect_url)
        current_page_title = f"'{page_owner_title_for_breadcrumbs}' এর পৃষ্ঠা নম্বর {to_bengali_numerals(page_data_row['page_number'])} সম্পাদনা"
        page_data = dict(page_data_row)
    else: # Adding a new page
        current_page_title = f"'{page_owner_title_for_breadcrumbs}' এর জন্য নতুন পৃষ্ঠা যোগ"
        page_data = None # Will be initialized later if it's a GET request for a new page

    if request.method == 'POST':
        page_content_html = request.form.get('page_content_html', '')
        page_number_str = request.form.get('page_number')
        page_number = 0 # Default

        if not page_number_str:
            flash('পৃষ্ঠা নম্বর আবশ্যক।', 'error')
        else:
            try:
                page_number = int(page_number_str)
                if page_number <= 0: raise ValueError("পৃষ্ঠা নম্বর ধনাত্মক হতে হবে।")
            except ValueError as e:
                flash(f'সঠিক পৃষ্ঠা নম্বর দিন: {e}', 'error')
                # Preserve entered data on error for form re-population
                page_data_on_error = {'id': page_id, 'page_content_html': page_content_html, 'page_number': page_number_str}
                return render_template('admin/manage_content_page_form.html', owner_type=owner_type, owner_id=owner_id, owner=owner_entity, page_data=page_data_on_error, page_id=page_id, current_page_title=current_page_title, form_action_url=request.url)

            # Check for uniqueness of page_number for the given owner, excluding the current page if editing
            query = 'SELECT id FROM content_pages WHERE owner_type = %s AND owner_id = %s AND page_number = %s'
            params = [owner_type, owner_id, page_number]
            if page_id:
                query += ' AND id != %s'
                params.append(page_id)

            existing_page_with_num = execute_query(query, tuple(params), fetchone=True)

            if existing_page_with_num:
                flash(f"পৃষ্ঠা নম্বর {to_bengali_numerals(page_number)} ইতিমধ্যে এই '{page_owner_title_for_breadcrumbs}' এর জন্য ব্যবহৃত হয়েছে।", 'error')
            else:
                if page_id: # Update existing page
                    execute_query(
                        'UPDATE content_pages SET page_content_html = %s, page_number = %s WHERE id = %s',
                        (page_content_html, page_number, page_id), commit=True
                    )
                    flash('পৃষ্ঠা সফলভাবে আপডেট করা হয়েছে।', 'success')
                else: # Insert new page
                    execute_query(
                        'INSERT INTO content_pages (owner_type, owner_id, page_number, page_content_html) VALUES (%s, %s, %s, %s)',
                        (owner_type, owner_id, page_number, page_content_html), commit=True
                    )
                    flash('নতুন পৃষ্ঠা সফলভাবে যোগ করা হয়েছে।', 'success')
                return redirect(redirect_url) # Redirect on successful save or update

        # If there was an error (e.g., empty page_number_str) and not redirected, re-render form with current data
        page_data_on_error = {'id': page_id, 'page_content_html': page_content_html, 'page_number': page_number_str if page_number_str else (page_data['page_number'] if page_data and 'page_number' in page_data else '')}
        return render_template('admin/manage_content_page_form.html', owner_type=owner_type, owner_id=owner_id, owner=owner_entity, page_data=page_data_on_error, page_id=page_id, current_page_title=current_page_title, form_action_url=request.url)

    # For GET request:
    if not page_data and not page_id: # If it's a GET request for a new page
        default_page_number = get_next_page_number(owner_type, owner_id)
        page_data = {'page_number': default_page_number, 'page_content_html': ''}

    # Determine form action URL
    form_action_url = url_for('manage_content_page', owner_type=owner_type, owner_id=owner_id, page_id=page_id) if page_id else url_for('manage_content_page', owner_type=owner_type, owner_id=owner_id)
    return render_template('admin/manage_content_page_form.html', owner_type=owner_type, owner_id=owner_id, owner=owner_entity, page_data=page_data, page_id=page_id, current_page_title=current_page_title, form_action_url=form_action_url)


@app.route('/admin/page/delete/<int:page_id>', methods=['POST'])
def delete_content_page(page_id):
    # Route to delete a specific content page.
    page_to_delete = get_content_page(page_id)
    if not page_to_delete:
        flash('ডিলিট করার জন্য পৃষ্ঠাটি খুঁজে পাওয়া যায়নি।', 'error')
        return redirect(url_for('admin_dashboard')) # Fallback redirect

    execute_query('DELETE FROM content_pages WHERE id = %s', (page_id,), commit=True)
    flash(f"পৃষ্ঠা নম্বর {to_bengali_numerals(page_to_delete['page_number'])} সফলভাবে ডিলিট করা হয়েছে।", 'success')

    # Redirect back to the owner's edit page or part management page
    if page_to_delete['owner_type'] == 'story':
        return redirect(url_for('edit_story', story_id=page_to_delete['owner_id']))
    elif page_to_delete['owner_type'] == 'story_part':
        return redirect(url_for('manage_story_part', part_id=page_to_delete['owner_id']))
    else:
        return redirect(url_for('admin_dashboard')) # Fallback


if __name__ == '__main__':
    # Main execution block
    # Database file check is no longer needed as PostgreSQL is a server-based database.
    # init_db() can be called once to ensure tables are created if they don't exist.
    try:
        # Test database connection first
        conn_test = get_db_connection()
        conn_test.close()
        # Attempt to display hostname, handle potential errors if DATABASE_URL format is unexpected
        try:
            db_host_info = DATABASE_URL.split('@')[1].split('/')[0]
        except IndexError:
            db_host_info = " (could not parse host from DATABASE_URL)"
        print(f"PostgreSQL ডাটাবেসের সাথে সংযোগ সফল: {db_host_info}")


        # In development, init_db() can be called on startup.
        # It will create tables if they don't exist (due to CREATE TABLE IF NOT EXISTS).
        # In production, this should ideally be a one-time manual step or part of deployment scripts.
        print("ডাটাবেস টেবিল এবং ট্রিগার পরীক্ষা/তৈরি করা হচ্ছে...")
        init_db() # Creates tables and triggers if they don't exist
        print("ডাটাবেস প্রস্তুত।")

    except psycopg2.OperationalError as e:
        print(f"PostgreSQL ডাটাবেসের সাথে সংযোগ স্থাপন করা যায়নি: {e}")
        print(f"অনুগ্রহ করে নিশ্চিত করুন ডাটাবেস সার্ভার চলছে এবং DATABASE_URL ({DATABASE_URL}) সঠিক আছে।")
        exit(1) # Exit the app if DB connection fails
    except Exception as e:
        print(f"একটি অপ্রত্যাশিত সমস্যা হয়েছে: {e}")
        exit(1) # Exit on other critical errors during startup

    app.run(host="0.0.0.0", port=5000, debug=True)
