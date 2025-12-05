from flask import Flask, request, render_template, send_from_directory
import os
import re
import docx2txt
import PyPDF2
import pickle
from flask_mail import Mail, Message
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Configure Flask-Mail
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME='abc@gmail.com',
    MAIL_PASSWORD='naqhsqkszmhcegwsddsssdu',  # App password, NOT regular account password
)

mail = Mail(app)

# Load pre-trained model and TF-IDF vectorizer and label encoder
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def read_pdf_text(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def read_docx_text(file_path):
    return docx2txt.process(file_path)

def extract_txt_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf_text(file_path)
    elif file_path.endswith('.docx'):
        return read_docx_text(file_path)
    elif file_path.endswith('.txt'):
        return extract_txt_text(file_path)
    else:
        return ""

def extract_emails(text):
    # Regex pattern to extract email addresses including domains with longer TLDs
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(pattern, text)
    return emails[0] if emails else None  # Return first found email or None

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

def send_email(subject, recipient, body):
    msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[recipient])
    msg.body = body
    mail.send(msg)

@app.route("/")
def matchresume():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        resume_categories = []
        candidate_emails = []

        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resume_text = extract_text(filename)
            resumes.append(resume_text)

            # Extract email from resume text
            email = extract_emails(resume_text)
            candidate_emails.append(email)

            category = pred(resume_text)
            resume_categories.append(category)

        if not resumes or not job_description:
            return render_template('index.html', message="Please upload resumes and enter a job description.")

        # Vectorize job description and resumes for similarity
        vectorizer = TfidfVectorizer().fit([job_description] + resumes)
        vectors = vectorizer.transform([job_description] + resumes).toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]
        top_categories = [resume_categories[i] for i in top_indices]
        top_emails = [candidate_emails[i] for i in top_indices]

        top_results = list(zip(top_resumes, similarity_scores, top_categories))

        # Send emails to top 5 candidates
        for i in range(len(top_emails)):
            email = top_emails[i]
            if email:
                subject = "Interview Invitation from HireSmart"
                body = (
                    f"Dear Candidate,\n\n"
                    f"This is a demonstration email generated as part of a Final Year Project (FYP). "
                    f"Please note that this is *not* an official or valid interview invitation.\n\n"
                    f"We are pleased to inform you that your profile has been used for testing purposes in the system's email notification feature.\n\n"
                    f"If you have received this message in error, kindly disregard it.\n\n"
                    f"Best regards,\n"
                    f"HireSmart Team\n"
                    f"(FYP Demo Email — Not an Official Communication)"
                )

                send_email(subject, email, body)

        return render_template(
            'index.html',
            message="Top matching resumes and invitation emails sent to candidates.",
            top_results=top_results
        )

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
