# database.py
import psycopg2
import psycopg2.extras # DictCursor এর জন্য
import os

# আপনার দেওয়া PostgreSQL URL
DATABASE_URL = "postgresql://golpokar_user:eRPrj5VkGUnYSmkD2abUo5DXZdKt1GBu@dpg-d0gs42juibrs73fu3gs0-a/golpokar"

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    # conn.row_factory = psycopg2.extras.DictCursor # কার্সার তৈরি করার সময় সেট করবো
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # গল্প/নোটের জন্য টেবিল (stories)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stories (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            has_parts BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # গল্পের বিভিন্ন অংশের জন্য টেবিল (story_parts)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS story_parts (
            id SERIAL PRIMARY KEY,
            story_id INTEGER NOT NULL,
            part_title TEXT NOT NULL,
            summary TEXT,
            part_order INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (story_id) REFERENCES stories (id) ON DELETE CASCADE
        )
    ''')

    # কনটেন্ট পৃষ্ঠা (content_pages)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_pages (
            id SERIAL PRIMARY KEY,
            owner_type TEXT NOT NULL CHECK(owner_type IN ('story', 'story_part')),
            owner_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            page_content_html TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (owner_type, owner_id, page_number)
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_content_pages_owner ON content_pages (owner_type, owner_id);
    ''')

    # --- ট্রিগার ফাংশন এবং ট্রিগার তৈরি (PostgreSQL এর জন্য) ---

    # updated_at কলাম স্বয়ংক্রিয়ভাবে আপডেট করার জন্য একটি ফাংশন
    cursor.execute('''
        CREATE OR REPLACE FUNCTION trigger_set_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
          NEW.updated_at = NOW();
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    ''')

    # stories টেবিলের জন্য ট্রিগার
    cursor.execute('''
        DROP TRIGGER IF EXISTS update_stories_updated_at_trigger ON stories;
        CREATE TRIGGER update_stories_updated_at_trigger
        BEFORE UPDATE ON stories
        FOR EACH ROW
        EXECUTE FUNCTION trigger_set_timestamp();
    ''')

    # story_parts টেবিলের জন্য ট্রিগার
    cursor.execute('''
        DROP TRIGGER IF EXISTS update_story_parts_updated_at_trigger ON story_parts;
        CREATE TRIGGER update_story_parts_updated_at_trigger
        BEFORE UPDATE ON story_parts
        FOR EACH ROW
        EXECUTE FUNCTION trigger_set_timestamp();
    ''')

    # content_pages টেবিলের জন্য ট্রিগার
    cursor.execute('''
        DROP TRIGGER IF EXISTS update_content_pages_updated_at_trigger ON content_pages;
        CREATE TRIGGER update_content_pages_updated_at_trigger
        BEFORE UPDATE ON content_pages
        FOR EACH ROW
        EXECUTE FUNCTION trigger_set_timestamp();
    ''')

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    print("ডাটাবেস ইনিশিয়ালাইজ করা হচ্ছে...")
    try:
        init_db()
        print("ডাটাবেস এবং টেবিল সফলভাবে ইনিশিয়ালাইজ হয়েছে (অথবা ইতিমধ্যে বিদ্যমান ছিলো)।")
    except Exception as e:
        print(f"ডাটাবেস ইনিশিয়ালাইজেশনে সমস্যা হয়েছে: {e}")
