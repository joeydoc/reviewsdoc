import streamlit as st
import pandas as pd
import re
from collections import Counter
import string
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Set visualization style
sns.set(style="whitegrid")

# Download NLTK tokenizer data, stopwords, and VADER lexicon
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')  # Added VADER lexicon download

download_nltk_data()
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Imported SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
stop_words.update([' ', '', '’', "´", "'"])  # Simplified adding multiple stopwords

# Define bad words
bad_words = [
    "awful", "terrible", "poor", "bad", "horrible", "worst", "disappointed",
    "unsatisfactory", "unprofessional", "rude", "incompetent", "frustrated",
    "disgusting", "slow", "unreliable", "messy", "dirty", "unfriendly",
    "annoying", "unpleasant", "disrespectful", "inconvenient", "uncomfortable",
    "expensive", "overpriced", "rip-off", "regret", "problematic", "subpar",
    "mediocre", "regretful", "confusing", "stressful", "insufficient",
    "disorganized", "delayed", "unhelpful", "noisy", "smelly", "pathetic",
    "lackluster", "bland", "frustrating", "nasty", "painful", "disheartened",
    "ineffective", "bitter", "infuriating", "broken", "careless", "defective",
    "dissatisfied", "disinterested", "harsh", "impolite", "misleading",
    "neglected", "offensive", "second-rate", "shabby", "substandard",
    "unacceptable", "underwhelming", "unfortunate", "worthless", "appalling",
    "irritating", "unresponsive", "abysmal", "exhausting", "cold", "obnoxious",
    "unbearable", "atrocious", "unappealing", "dreary", "bothersome", "unfair",
    "monotonous", "outdated", "unworthy", "disturbing", "grimy"
]

# Define good words
good_words = [
    "good", "great", "excellent", "amazing", "fantastic", "positive",
    "pleasant", "happy", "satisfied", "love", "wonderful", "best",
    "superb", "delightful", "enjoyed", "friendly", "helpful", "efficient",
    "reliable", "clean", "comfortable", "quick", "reasonable", "nice",
    "perfect", "lovely", "brilliant", "impressive", "outstanding",
    "exceptional", "awesome", "marvelous", "cheerful", "genius",
    "sparkling", "stellar", "remarkable", "sincere", "charming"
]

# Define medical keywords
medical_keywords = [
    "Colonoscopy",
    "Gastroscopy",
    "Gastroenterology",
    "Colonoscopy,Gastroscopy",
    "Knee Replacement",
    "Cataract",
    "Hip Replacement",
    "Cataract Surgery",
    "Hernia",
    "Gastric Sleeve",
    "Gastroscopy,Colonoscopy",
    "Arthroscopy",
    "Colorectal Surgery",
    "Endoscopy",
    "Bladder Cancer",
    "Weight Loss",
    "Gynaecology",
    "Hysteroscopy",
    "Shoulder Pain",
    "Gallbladder Removal (Cholecystectomy)",
    "Knee Arthroscopy",
    "Weight Loss & Bariatric Surgery",
    "General Dentistry",
    "Fracture (Broken Bone)",
    "Meniscus Tear Surgery",
    "Open Hernia Repair",
    "Tooth Extraction",
    "Laparoscopy",
    "Cataracts",
    "Prostate Cancer",
    "Inguinal Hernia",
    "Prostate Biopsy",
    "Total Knee Replacement",
    "Hernia Repair (Keyhole)",
    "Colonoscopy,Endoscopy",
    "Hand & Wrist Surgery",
    "Iron Deficiency Anaemia",
    "Hernia,Open Hernia Repair",
    "Hysterectomy",
    "Dental Surgery",
    "Upper GI (Gastrointestinal) Surgery",
    "Arthroscopy,Meniscus Tear Surgery",
    "Knee Pain",
    "Oral Surgery",
    "Acid Reflux",
    "Orthodontics",
    "Paediatric (Pediatric) Dentistry",
    "Rheumatology",
    "Skin Cancer",
    "Arthritis",
    "Carpal Tunnel Decompression",
    "Colposcopy",
    "Nephrology (Renal Medicine)",
    "Gastroscopy,Endoscopy",
    "Skin Lesion Removal",
    "Tendon Repair",
    "Basal Cell Carcinoma",
    "Benign Prostate Hyperplasia",
    "Carpal Tunnel Syndrome",
    "Colonic Polyps",
    "Endocrine Surgery",
    "Fibroids",
    "Gallstones",
    "Inguinal Hernia,Open Hernia Repair",
    "Knee Arthroscopy,Meniscus Tear Surgery",
    "Respiratory Medicine",
    "Vaginal Prolapse",
    "Weight Loss Surgery",
    "ACL Reconstruction",
    "Adult Ear, Nose & Throat Surgery",
    "Cataracts,Cataract Surgery",
    "Endometriosis",
    "Gastroenterology,Colonoscopy",
    "Gastroenterology,Colorectal Surgery",
    "Hernia,Inguinal Hernia",
    "Sleep Disorders",
    "Thyroidectomy",
    "Abdominoplasty (Tummy Tuck)",
    "Anterior Hip Replacement",
    "Arthroscopy,Shoulder Pain",
    "Back Pain",
    "Big Toe Fusion Surgery",
    "Blood in Urine (Haematuria)",
    "Colonoscopy,Colonic Polyps",
    "Coloproctology",
    "Cystoscopy",
    "Direct Anterior Approach (DAA) Hip Replacement",
    "Gastroscopy,Acid Reflux",
    "Heavy Periods (Menorrhagia)",
    "Laparoscopic Hysterectomy",
    "Reconstructive Surgery",
    "Robotic Assisted Knee Surgery",
    "Shoulder Arthroscopy",
    "Shoulder Pain,Arthroscopy",
    "Video Consultation",
    "Ankle Arthroscopy",
    "Arthroscopy,Knee Arthroscopy",
    "Breast Cancer",
    "Breast Surgery",
    "Bunionette Deformity Correction",
    "Cataract Surgery,Cataracts",
    "Colon Cancer",
    "Colonoscopy,Colitis (Ulcerative Colitis)",
    "Colonoscopy,Gastroscopy,Endoscopy",
    "Colonoscopy,Gastroscopy,Iron Deficiency Anaemia",
    "Endoscopy,Colonoscopy",
    "Facial Plastic Reconstructive Surgery",
    "Gastric Banding",
    "Gastric Bypass",
    "Hernia,Gastric Sleeve",
    "Hernia,Inguinal Hernia,Open Hernia Repair",
    "Knee Replacement,Total Knee Replacement",
    "Laparoscopy,Gastroscopy",
    "Mastectomy",
    "Meniscus Tear Surgery,Knee Arthroscopy",
    "PSA",
    "Tooth Extraction,Oral Surgery",
    "Tooth Extraction,Paediatric (Pediatric) Dentistry",
    "Urinary Incontinence",
    "Acid Reflux,Gastroscopy",
    "Arthroscopy,Knee Arthroscopy,Meniscus Tear Surgery",
    "Bladder Cancer,Prostate Cancer",
    "Bunion Surgery",
    "Colonoscopy,Bowel Cancer",
    "Colonoscopy,Coeliac Disease",
    "Colonoscopy,Colposcopy",
    "Colonoscopy,Crohn's Disease",
    "Colonoscopy,Endoscopy,Gastroscopy",
    "Colonoscopy,Gastroenterology",
    "Coloproctology,Colonoscopy",
    "Direct Anterior Approach (DAA) Hip Replacement ,Hip Replacement",
    "Endometriosis,Hysteroscopy",
    "Endoscopy,Acid Reflux",
    "Food Intolerance",
    "Fundoplication",
    "Gallbladder Removal (Cholecystectomy),Hernia Repair (Keyhole)",
    "Gallstones,Laparoscopy",
    "Ganglion Cyst Removal",
    "Gastro-oesophageal Reflux Disease (GORD)",
    "Gastroscopy,Colonoscopy,Iron Deficiency Anaemia",
    "Gastroscopy,Irritable Bowel Syndrome (IBS)",
    "Hernia,Hernia Repair (Keyhole)",
    "Hernia,Umbilical (Belly-Button) Hernia",
    "Hip and Groin Pain",
    "Hip Replacement,Direct Anterior Approach (DAA) Hip Replacement",
    "Hydrocelectomy",
    "Hysterectomy,Fibroids",
    "Iron Deficiency Anaemia,Colonoscopy",
    "Lung Infection or Chest Infection",
    "Medical Report",
    "Meniscus Tear Surgery,Arthroscopy",
    "Minimally Invasive Foot & Ankle Surgery",
    "Minimally Invasive Knee Surgery",
    "Nephrectomy (Laparoscopic)",
    "Obstructive Sleep Apnoea (OSA)",
    "Open Hernia Repair,Hernia",
    "Open Hernia Repair,Inguinal Hernia",
    "Paediatric (Pediatric) Ear, Nose & Throat Surgery",
    "Prostate Cancer Diagnostics",
    "Prostate Cancer,Bladder Cancer",
    "Prostatectomy (TURP)",
    "Reproductive Medicine",
    "Revision Surgery",
    "Rotator Cuff Repair",
    "Shoulder Pain,Tendon Repair",
    "Shoulder Replacement",
    "Skin Cancer,Basal Cell Carcinoma",
    "Thyroid Surgery",
    "Tooth Extraction,Oral Surgery,Dental Surgery",
    "Urological Cancers",
    "Wisdom Teeth Removal",
    "Achilles Tendon Reconstruction",
    "Allergic Rhinitis (Hayfever),Nasal Blockage",
    "Anal Fissure",
    "Ankle Arthroscopy,Arthroscopy",
    "Anterior Hip Replacement,Direct Anterior Approach (DAA) Hip Replacement ,Hip Replacement",
    "Appendicectomy",
    "Arthritis (Foot)",
    "Arthritis (Shoulder),Dislocated Shoulder",
    "Arthritis,Rheumatology",
    "Arthroscopy,Knee Pain,Knee Arthroscopy",
    "Arthroscopy,Knee Pain,Meniscus Tear Surgery",
    "Arthroscopy,Meniscus Tear Surgery,Knee Arthroscopy",
    "Atrial Fibrillation",
    "Back Pain,Arthritis",
    "Basal Cell Carcinoma,Mole Removal",
    "Basal Cell Carcinoma,Skin Cancer,Facial Plastic Reconstructive Surgery",
    "Blood in Urine (Haematuria),Kidney Disease Diagnosis & Treatment",
    "Blue Light Cystoscopy",
    "Bowel Cancer Screening,Irritable Bowel Syndrome (IBS),Colonoscopy",
    "Bowel Cancer,Gastroscopy,Colonoscopy",
    "Breast Lump",
    "Bunion (Hallux Valgus) Correction,Bunionette Deformity Correction",
    "Bunion Surgery,Bunionette Deformity Correction",
    "Bunionectomy",
    "Cataract Surgery,Floaters (Eye)",
    "Chest Pain,Anaemia,Back Pain",
    "Colon Cancer Screening",
    "Colon Cancer,Colonic Polyps",
    "Colonoscopy,Acid Reflux,Gastroscopy",
    "Colonoscopy,Bowel Cancer Screening",
    "Colonoscopy,Endoscopy,Iron Deficiency Anaemia",
    "Colonoscopy,Gastroscopy,Barrett's Oesophagus",
    "Colonoscopy,Gastroscopy,Capsule Endoscopy",
    "Colonoscopy,Gastroscopy,Colitis (Ulcerative Colitis)",
    "Colonoscopy,Gastroscopy,Gastroenterology",
    "Colonoscopy,Gastroscopy,Irritable Bowel Syndrome (IBS)",
    "Colonoscopy,Removal of Rectum and Colon (Proctocolectomy)",
    "Colorectal Surgery,Gastroenterology",
    "Colposcopy,Colonoscopy,Coloproctology",
    "Colposcopy,Gastroscopy",
    "COVID-19 - Rehabilitation and Recovery",
    "Cysts,Basal Cell Carcinoma",
    "Cysts,Fibroids,Heavy Periods (Menorrhagia)",
    "Cysts,Ovarian Cysts",
    "Endocrine Surgery,Endodontics",
    "Endoscopic Management of Gastrointestinal Tract Tumours and Polyps",
    "Endoscopy,Acid Reflux,Colonoscopy",
    "Fibroids,Hysteroscopy",
    "Finger Injury",
    "Floaters (Eye) ,Cataracts,Cataract Surgery",
    "Food Intolerance,Constipation,Colonoscopy",
    "Fracture (Broken Bone),Hand & Wrist Surgery",
    "Fundoplication,Laparoscopy,Hernia Repair (Keyhole)",
    "Gallbladder Removal (Cholecystectomy),Gallstones",
    "Gallbladder Removal (Cholecystectomy),Laparoscopy",
    "Gallstones,Gallbladder Removal (Cholecystectomy)",
    "Gallstones,Laparoscopy,Gallbladder Removal (Cholecystectomy)",
    "Ganglion Cyst Removal,Cysts,Hand Surgery",
    "Gastric Bypass,Gastric Sleeve",
    "Gastric Sleeve,Gastroscopy",
    "Gastric Sleeve,Hernia Repair (Keyhole)",
    "Gastritis",
    "Gastroenterology,Endodontics",
    "Gastroenterology,General Dentistry",
    "Gastroscopy,Colonoscopy,Coeliac Disease",
    "Gastroscopy,Colonoscopy,Colitis (Ulcerative Colitis)",
    "Gastroscopy,Colonoscopy,Endoscopy",
    "Gastroscopy,Gastro-oesophageal Reflux Disease (GORD)",
    "Gastroscopy,Hernia Repair (Keyhole)",
    "Gastroscopy,Inflammatory Bowel Disease",
    "Gastroscopy,Iron Deficiency Anaemia,Colonoscopy",
    "Gastroscopy,Liver Disease",
    "Gastroscopy,Weight Loss Surgery",
    "Haemorrhoids",
    "Hammer Toes",
    "Heart Attack",
    "Hepatology,Gastroenterology",
    "Hernia Repair (Keyhole),Hernia",
    "Hernia Repair (Keyhole),Inguinal Hernia,Hernia",
    "Hernia Repair (Keyhole),Laparoscopy,Gastroscopy",
    "Hernia,Gastric Banding",
    "Hernia,Gastroscopy",
    "Hernia,Open Hernia Repair,Inguinal Hernia",
    "Herniated Disc or Slipped Disc,Revision Surgery,Fracture (Broken Bone)",
    "Hiatus Hernia,Open Hernia Repair",
    "High Blood Pressure,Echocardiogram",
    "Hip Arthroscopy",
    "Hip Preservation Surgery",
    "Hip Replacement,Tendon Repair,Hand & Wrist Surgery",
    "Humerus Fractures",
    "Hysterectomy,Laparoscopic Hysterectomy",
    "Hysterectomy,Vaginal Prolapse",
    "Hysteroscopy,Endometriosis",
    "Hysteroscopy,Fibroids",
    "Hysteroscopy,Heavy Periods (Menorrhagia)",
    "Hysteroscopy,Laparoscopic Hysterectomy",
    "Inflammatory Bowel Disease,Colonoscopy,Colitis (Ulcerative Colitis)",
    "Inflammatory Bowel Disease,Colonoscopy,Endoscopy",
    "Inguinal Hernia,Gastric Sleeve",
    "Inguinal Hernia,Open Hernia Repair,Hernia",
    "Inguinal Hernia,Umbilical (Belly-Button) Hernia",
    "Iron Deficiency Anaemia,Gastroscopy,Colonoscopy",
    "Irregular Periods",
    "Kidney Stone Prevention",
    "Knee Arthroscopy,Fracture (Broken Bone),Meniscus Tear Surgery",
    "Knee Arthroscopy,Knee Cartilage Surgery",
    "Knee Arthroscopy,Meniscus Tear Surgery,Ankle Joint Fusion",
    "Knee Ligament Repair,Knee Pain",
    "Knee Pain,Arthroscopy",
    "Knee Pain,Fracture (Broken Bone)",
    "Knee Pain,Hip Replacement,Hip and Groin Pain",
    "Knee Pain,Knee Cyst",
    "Knee Replacement,Fracture (Broken Bone)",
    "Knee Replacement,Hip Replacement",
    "Knee Replacement,Partial Knee Replacement",
    "Knee Replacement,Robotic Assisted Knee Surgery",
    "Laparoscopic Hysterectomy,Endometriosis,Adenomyosis",
    "Laparoscopy,Gastric Sleeve",
    "Lipoma Removal",
    "Liver Disease,Colorectal Surgery",
    "Meniscus Tear Surgery,Shoulder Pain,Arthroscopy",
    "Mole Removal,Skin Tag Removal",
    "Neck Pain,Back Pain,Shoulder Pain",
    "Nephrectomy (Laparoscopic),Prostatectomy (Robotic),Male Infertility",
    "Oral Surgery,Tooth Extraction",
    "Orthodontics,Cardiology",
    "Osteoarthritis,Arthritis",
    "Palliative Care",
    "Parathyroidectomy",
    "Periodontics,General Dentistry",
    "Periodontics,Orthodontics",
    "Postoperative Rehabilitation",
    "Prostate Biopsy,Prostate Cancer",
    "Prostate Biopsy,Urinary Incontinence",
    "Prostate Cancer Diagnostics,Prostate Cancer",
    "Prostate Cancer,Prostate Biopsy,Bladder Cancer",
    "Prostate Cancer,Prostate Biopsy,PSA",
    "Prostate Cancer,PSA,Prostate Biopsy",
    "Reproductive Medicine,Gynaecology",
    "Respiratory Medicine,Rheumatology",
    "Rheumatology,Arthritis",
    "Rheumatology,Respiratory Medicine",
    "Robotic Assisted Knee Surgery,Knee Replacement",
    "Robotic Assisted Partial Knee Replacement",
    "Shoulder Arthroscopy,Shoulder Pain,Rotator Cuff Repair",
    "Skin Cancer Surgery",
    "Skin Cancer,Basal Cell Carcinoma,Facial Plastic Reconstructive Surgery",
    "Skin Lesion Removal,Skin Cancer,Skin Blushing",
    "Sleep Disorders,Obstructive Sleep Apnoea (OSA)",
    "Thyroid Problems,Thyroid Surgery",
    "Thyroidectomy,Thyroid Nodules,Thyroid Problems",
    "Tooth Extraction,Crowns",
    "Tooth Extraction,Dental Surgery,Paediatric (Pediatric) Dentistry",
    "Tooth Extraction,Paediatric (Pediatric) Dentistry,Dental Surgery",
    "Transperineal Prostate Biopsy under Local Anaesthesia",
    "Trigger Finger",
    "Troublesome Cough,Lung Infection or Chest Infection",
    "Umbilical (Belly-Button) Hernia,Hernia,Hernia Repair (Keyhole)",
    "Umbilical (Belly-Button) Hernia,Inguinal Hernia",
    "Umbilical (Belly-Button) Hernia,Open Hernia Repair",
    "Upper Gastrointestinal Conditions (Oesophagus & Stomach),Bowel Cancer Screening,Colonoscopy",
    "Upper GI (Gastrointestinal) Surgery,Gastroenterology",
    "Upper GI (Gastrointestinal) Surgery,Weight Loss & Bariatric Surgery",
    "Uro-Gynaecology",
    "Urology",
    "Vaginal Prolapse,Hysterectomy,Hysteroscopy",
    "Vasectomy (Male Sterilization)",
    "Video Consultation,Colonoscopy",
    "Weight Loss & Bariatric Surgery,Gastroenterology",
    "Weight Loss Surgery,Gastric Sleeve",
    "Wisdom Teeth Removal,Impacted/Infected Wisdom Teeth Removal"
]

# Preprocess medical keywords for efficient searching
medical_keywords_lower = [kw.lower() for kw in medical_keywords]

# Create a mapping from lowercase to original casing for medical keywords
medical_keywords_lower_dict = {kw.lower(): kw for kw in medical_keywords}

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment score using VADER
def calculate_sentiment_score(review_text):
    if not isinstance(review_text, str):
        review_text = str(review_text)
    scores = sid.polarity_scores(review_text)
    # Normalize compound score to 0-100
    sentiment_score = round((scores['compound'] + 1) * 50, 2)
    return sentiment_score

# Streamlit App Layout
st.title("Sentiment Analysis and Medical Keyword Extraction")
st.write("Upload a CSV or Excel file containing a 'Review' column to analyze sentiment and extract medical keywords.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Determine file type
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Read the file into a DataFrame
    try:
        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        st.success(f"Successfully loaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()
    
    df = df.dropna(subset=['Review'])
    df = df[df['Review'].astype(str).str.strip() != '']

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Display DataFrame information
    st.subheader("Data Overview")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    columns_str = ", ".join(df.columns.tolist())
    st.write(f"Column Names: {columns_str}")
    
    # Check for 'Review' column
    if 'Review' not in df.columns:
        st.error("Error: The uploaded file does not contain a 'Review' column.")
        st.stop()
    else:
        st.subheader("Sample Reviews")
        st.write(df['Review'].head())

    # Button to process the data
    if st.button("Process Data"):
        with st.spinner("Processing..."):
            # Initialize counters
            total_good_word_counter = Counter()
            total_bad_word_counter = Counter()
            total_word_counter = Counter()
            keyword_counter = Counter()

            # Initialize lists to store new column data
            sentiment_scores = []
            good_word_counts = []
            bad_word_counts = []
            good_words_found = []
            bad_words_found = []

            # Prepare translation table for removing punctuation
            translator = str.maketrans('', '', string.punctuation)

            # Iterate through each review
            for index, review in df['Review'].items():
                if not isinstance(review, str):
                    review = str(review)

                # Calculate Sentiment Score using VADER
                sentiment_score = calculate_sentiment_score(review)
                sentiment_scores.append(sentiment_score)

                # Convert to lowercase
                review_lower = review.lower()

                # Tokenize the review
                tokens = word_tokenize(review_lower)

                # Remove punctuation from tokens
                tokens = [word.translate(translator) for word in tokens if word]

                # Filter out stop words
                tokens = [word for word in tokens if word not in stop_words]

                # Count good and bad words in the review
                review_good_words = [word for word in tokens if word in good_words]
                review_bad_words = [word for word in tokens if word in bad_words]

                good_count = len(review_good_words)
                bad_count = len(review_bad_words)

                # Update total counters
                total_good_word_counter.update(review_good_words)
                total_bad_word_counter.update(review_bad_words)
                total_word_counter.update(tokens)

                # Find unique good and bad words in the review
                unique_good = list(set(review_good_words))
                unique_bad = list(set(review_bad_words))

                # Append to lists
                good_word_counts.append(good_count)
                bad_word_counts.append(bad_count)
                good_words_found.append(", ".join(unique_good) if unique_good else "")
                bad_words_found.append(", ".join(unique_bad) if unique_bad else "")

                # Keyword counting (exact match)
                if 'Keywords' in df.columns:
                    keywords_entry = df.at[index, 'Keywords']
                    if not isinstance(keywords_entry, str):
                        keywords_entry = str(keywords_entry)

                    # Split the keywords by comma or semicolon
                    keywords_split = re.split(r'[;,]', keywords_entry)
                    keywords_cleaned = [kw.strip().lower() for kw in keywords_split if kw.strip()]

                    # Count each keyword
                    for kw in keywords_cleaned:
                        if kw in medical_keywords_lower_dict:
                            keyword_counter[kw] += 1

            # Add new columns to the DataFrame
            df['Sentiment_Score'] = sentiment_scores
            df['Good_Word_Count'] = good_word_counts
            df['Bad_Word_Count'] = bad_word_counts
            df['Good_Words_Found'] = good_words_found
            df['Bad_Words_Found'] = bad_words_found

            # Visualization: Pie Chart for Positive and Negative Words
            total_good = sum(total_good_word_counter.values())
            total_bad = sum(total_bad_word_counter.values())

            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.pie([total_good, total_bad], labels=['Positive Words', 'Negative Words'],
                    autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'], explode=(0.1, 0))
            ax1.set_title('Percentage of Positive and Negative Words')
            ax1.axis('equal')

            st.pyplot(fig1)

            # Visualization: Bar Chart for Top 25 Most Used Words
            filtered_word_counter = Counter({word: count for word, count in total_word_counter.items() if word not in stop_words})
            top_25_words = filtered_word_counter.most_common(25)
            words, counts = zip(*top_25_words) if top_25_words else ([], [])

            fig2, ax2 = plt.subplots(figsize=(12, 8))
            sns.barplot(x=list(counts), y=list(words), palette='viridis', ax=ax2)
            ax2.set_title('Top 25 Most Used Words (Excluding Stop Words)')
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Words')
            plt.tight_layout()

            st.pyplot(fig2)

            # Visualization: Bar Chart for Top 15 Medical Keywords
            if 'Keywords' in df.columns:
                # Get top 15 medical keywords
                top_15_keywords = keyword_counter.most_common(15)
                if top_15_keywords:
                    keywords, keyword_counts = zip(*top_15_keywords)

                    # Convert keywords back to original casing
                    keywords_original = [medical_keywords_lower_dict.get(kw, kw) for kw in keywords]

                    # Create bar chart
                    fig3, ax3 = plt.subplots(figsize=(14, 8))
                    sns.barplot(x=list(keyword_counts), y=list(keywords_original), palette='magma', ax=ax3)
                    ax3.set_title('Top 15 Medical Keywords Mentioned (Using Keywords Column)')
                    ax3.set_xlabel('Count')
                    ax3.set_ylabel('Medical Keywords')
                    plt.tight_layout()

                    st.pyplot(fig3)
                else:
                    st.warning("No medical keywords found in the 'Keywords' column.")
            else:
                st.info("The uploaded file does not contain a 'Keywords' column. Skipping medical keywords visualization.")

            # Display All Bad Words and Their Counts
            bad_words_df = pd.DataFrame(total_bad_word_counter.items(), columns=['Bad Word', 'Count']).sort_values(by='Count', ascending=False)
            st.subheader("All Bad Words and Their Counts")
            st.write(bad_words_df)

            # Display All Good Words and Their Counts
            good_words_df = pd.DataFrame(total_good_word_counter.items(), columns=['Good Word', 'Count']).sort_values(by='Count', ascending=False)
            st.subheader("All Good Words and Their Counts")
            st.write(good_words_df)

            # Provide Downloadable Data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed Data as CSV",
                data=csv,
                file_name='processed_data.csv',
                mime='text/csv',
            )

            # Optionally, provide downloads for the plots
            # Function to save matplotlib figure to bytes
            def fig_to_bytes(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                return buf.read()

            # Download Pie Chart
            pie_bytes = fig_to_bytes(fig1)
            st.download_button(
                label="Download Pie Chart",
                data=pie_bytes,
                file_name='pie_chart.png',
                mime='image/png',
            )

            # Download Top 25 Words Bar Chart
            bar25_bytes = fig_to_bytes(fig2)
            st.download_button(
                label="Download Top 25 Words Bar Chart",
                data=bar25_bytes,
                file_name='top_25_words_bar_chart.png',
                mime='image/png',
            )

            # Download Top 15 Medical Keywords Bar Chart if it exists
            if 'Keywords' in df.columns and 'fig3' in locals():
                bar15_bytes = fig_to_bytes(fig3)
                st.download_button(
                    label="Download Top 15 Medical Keywords Bar Chart",
                    data=bar15_bytes,
                    file_name='top_15_medical_keywords_bar_chart.png',
                    mime='image/png',
                )
else:
    st.info("Awaiting for CSV or Excel file to be uploaded.")
