# Project Report: AI-Powered Course Recommender System

**Author/Developer:** Suparn Nayak  
**Project Type:** Machine Learning / Web Application  
**Technologies Used:** Python, Flask, Pandas, Scikit-learn, HTML5, CSS3  

---

## 1. Executive Summary
The Course Recommender System is an intelligent web application designed to help users find the best courses on Coursera tailored to their interests. By entering a topic or specific skills (e.g., "Machine Learning", "Data Science"), the system utilizes Natural Language Processing (NLP) techniques to suggest the top 10 most relevant courses. The project recently underwent significant architectural modernization to improve response times, enhance security, and deliver a premium user interface.

## 2. Dataset Overview
The underlying dataset (`Coursera.csv`) contains metadata for over 3,500 courses. Key features extracted and utilized by the system include:
- **Course Name**: The official title of the course.
- **University**: The institution providing the course.
- **Difficulty Level**: Beginner, Intermediate, or Advanced.
- **Course Rating**: Average user rating.
- **Course URL**: Direct link to the Coursera page.
- **Skills**: A text-heavy column detailing the specific skills acquired in the course. 

*Note: The recommendation engine primarily relies on the `Skills` column to determine the semantic similarity between courses.*

## 3. Core Recommendation Algorithm
The recommendation engine is built on content-based filtering. The workflow is as follows:

1. **Text Vectorization (TF-IDF):** 
   The `Skills` associated with each course are processed using Scikit-Learn's `TfidfVectorizer`. This converts the textual data into numerical vectors, assigning higher weights to unique, defining words (like specific programming languages) and penalizing common stop words. The model analyzes n-grams ranging from 1 to 3 words.
2. **Similarity Measurement (Sigmoid Kernel):**
   To determine how closely related two courses are, the system computes the pairwise similarity between their TF-IDF vectors using a `sigmoid_kernel`. This generates a matrix where each course has a similarity score relative to every other course.
3. **Fuzzy Searching:**
   When a user inputs a query, `difflib.get_close_matches` is used to find the closest matching course name in the dataset.
4. **Ranking:**
   The system retrieves the index of the matched course, sorts the similarity scores in descending order, and returns the top 10 most similar courses.

## 4. System Architecture & Modernization
The project was recently refactored to align with modern software engineering standards. Key improvements include:

### A. Performance Optimization
- **Global Model Precomputation:** Originally, the application loaded the CSV and recalculated the massive TF-IDF similarity matrix on every single user request. This caused significant lag. The matrix calculation was moved to the application's initialization phase. It now loads only once when the server starts, reducing query response times from several seconds to a fraction of a second.
- **Duplicate Handling:** Fixed a critical bug where duplicate course names in the dataset caused Pandas to return a `Series` instead of an integer index, leading to a `ValueError` during sorting.

### B. Security Enhancements
- **XSS Prevention:** The legacy application converted raw Pandas DataFrames to HTML and marked them as safe for rendering. This posed a Cross-Site Scripting (XSS) vulnerability. The updated architecture passes raw data structures to Jinja2, which automatically escapes HTML characters, ensuring malicious input cannot be executed in the browser.

### C. UI/UX Redesign
- **Vanilla CSS Glassmorphism:** The outdated Materialize CSS framework was replaced entirely with a custom, modern design. 
- **Dark Mode Aesthetics:** Implemented a sleek dark theme featuring gradient backgrounds, floating abstract shapes, and translucent glass-like cards.
- **Responsive Grid:** Results are now displayed as highly readable, responsive cards featuring badges for Difficulty and Rating, greatly improving the user experience over the legacy table view.

## 5. Setup and Execution
To run the application locally:

1. Ensure Python is installed along with the packages listed in `requirements.txt` (Flask, Pandas, Scikit-learn, etc.).
2. Navigate to the project directory:
   ```bash
   cd d:\Ml_project\course-recommender
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Access the application in a web browser at `http://127.0.0.1:5000`.

---
*End of Report*
