import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import difflib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes so Vercel can talk to Render
CORS(app)

# --- Initialization: Load and Precompute Model Data ---
try:
    # Read dataset
    df_org = pd.read_csv('Coursera.csv')
    df = df_org.copy()
    
    # We only need 'Skills' and 'Course Name' for the recommendation model
    df['cleaned'] = df['Skills'].fillna('')
    
    # Fit TF-IDF Vectorizer
    tfv = TfidfVectorizer(
        min_df=3, max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), stop_words='english'
    )
    tfv_matrix = tfv.fit_transform(df['cleaned'])
    
    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    
    # Reverse mapping of indices and titles
    indices = pd.Series(df.index, index=df['Course Name'])
    # Ensure no duplicate indices by keeping first
    indices = indices[~indices.index.duplicated(keep='first')]
    
    # Prepare list of course names for difflib
    course_namelist = df['Course Name'].tolist()
    
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False


def give_rec(title, sig=sig):
    if not model_loaded:
        return pd.DataFrame()
        
    try:
        # Get the index corresponding to original_title
        idx = indices[title]

        # Get the pairwise similarity scores 
        sig_scores = list(enumerate(sig[idx]))

        # Sort the courses
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

        # Scores of the 10 most similar courses (excluding the first one, which is the course itself)
        sig_scores = sig_scores[1:11]

        # Course indices
        course_indices = [i[0] for i in sig_scores]

        # Top 10 most similar courses
        return df_org.iloc[course_indices]
    except KeyError:
        return pd.DataFrame()

def create_sim(search):
    if not search or not model_loaded:
        return pd.DataFrame()
        
    simlist = difflib.get_close_matches(search, course_namelist)
    if not simlist:
        return pd.DataFrame()
        
    # Get recommendations based on the closest match
    findf = give_rec(simlist[0])
    findf = findf.reset_index(drop=True)
    return findf

# --- API Routes ---

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    
    if not data or 'course' not in data:
        return jsonify({
            'success': False,
            'message': 'Please provide a course name.'
        }), 400
        
    namec = data['course'].strip()
    
    if not namec:
        return jsonify({
            'success': False,
            'message': 'Course name cannot be empty.'
        }), 400
        
    output = create_sim(namec)
    
    if output.empty:
        return jsonify({
            'success': True,
            'message': 'Sorry! We did not find any matching courses. Try adding more keywords in your search.',
            'courses': []
        })
    else:
        courses = output.to_dict('records')
        return jsonify({
            'success': True,
            'message': 'Here are some recommendations based on your search:',
            'courses': courses
        })

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Course Recommender API is running. Use POST /api/predict to get recommendations."
    })

if __name__ == '__main__':
    # Use environment variable for debug mode, default to False for production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))