from flask import Flask, request, render_template, jsonify
import spacy
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

# Dummy correct answer for comparison
correct_answer = """
The internet is a global network of computers that allows people to share information and communicate.
"""

# Text corpus for plagiarism detection
corpus = [
    "The internet is a global network...",
    "Communication is made easier through internet connectivity..."
]

def evaluate_grammar(text):
    doc = nlp(text)
    errors = 0
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ != "ROOT":
            errors += 1
    return errors

def detect_plagiarism(student_answer):
    vectorizer = TfidfVectorizer().fit_transform(corpus + [student_answer])
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[-1][:-1]  # Compare last element (student answer) to the rest
    max_similarity = max(similarity_scores) * 100
    return max_similarity

def check_accuracy(student_answer):
    ratio = difflib.SequenceMatcher(None, student_answer, correct_answer).ratio()
    return ratio * 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    student_answer = request.form['answer']
    
    # Accuracy based on similarity to model answer
    accuracy = check_accuracy(student_answer)
    
    # Grammar and coherence (simple error counting)
    grammar_errors = evaluate_grammar(student_answer)
    
    # Plagiarism detection
    plagiarism_score = detect_plagiarism(student_answer)
    
    # Final score (weighted)
    final_score = (accuracy * 0.7) - (grammar_errors * 0.1) - (plagiarism_score * 0.2)
    
    result = {
        "accuracy": accuracy,
        "grammar_errors": grammar_errors,
        "plagiarism_score": plagiarism_score,
        "final_score": final_score
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
