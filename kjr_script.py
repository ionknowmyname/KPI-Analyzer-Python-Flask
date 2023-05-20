import os
import argparse
import PyPDF2
from nltk.tokenize import sent_tokenize
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, send_file
from cloudinary.uploader import upload
import cloudinary.api
import dotenv


dotenv.load_dotenv()

app = Flask(__name__)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text += page.extract_text()
        return text.lower()

def preprocess_text(text):
    text = text.replace('•', '-')
    preprocessed_sentences = sent_tokenize(text)
    return preprocessed_sentences

def compare_tasks(job_description, report):
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_report = preprocess_text(report)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    encoded_job_description = tokenizer(preprocessed_job_description, padding=True, truncation=True, return_tensors='pt')
    encoded_report = tokenizer(preprocessed_report, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        job_description_embeddings = model(**encoded_job_description).last_hidden_state[:, 0, :]
        report_embeddings = model(**encoded_report).last_hidden_state[:, 0, :]
    
    similarity_matrix = cosine_similarity(job_description_embeddings, report_embeddings)
    matching_sentence_indices = similarity_matrix.argsort()[0][::-1]
    
    similarity_threshold = 0.7
    
    matching_sentences = []
    for index in matching_sentence_indices:
        similarity_score = similarity_matrix[0][index]
        if similarity_score >= similarity_threshold:
            matching_sentences.append(preprocessed_report[index])
    
    keywords = {'successful', 'completed', 'concluded', 'agreed', 'developed', 'monitored', 'implemented', 'created', 'coordinated', 'organized'}
    
    scores = []
    for sentence in matching_sentences:
        score = '50%'
        for word in keywords:
            if word in sentence:
                score = '100%'
        scores.append([sentence, score])

    return scores

@app.route('/evaluate', methods=['POST'])
def extract_matching_sentences():
    # Get the uploaded files from the request
    report_file = request.files.get('report')
    responsibility_file = request.files.get('responsibility')
    
    # Save the uploaded files to temporary locations
    report_path = '/tmp/report.pdf'
    responsibility_path = '/tmp/responsibility.pdf'
    report_file.save(report_path)
    responsibility_file.save(responsibility_path)
    
    # Extract text from the PDF files
    job_description_text = extract_text_from_pdf(responsibility_path)
    report_text = extract_text_from_pdf(report_path)
    
    # Compare tasks and find the matching sentences
    matching_sentences = compare_tasks(job_description_text, report_text)
    
    # Create a PDF file with the matching sentences
    result_file = '/tmp/kjr_result.docx'
    
    with open(result_file, 'w') as file:
        file.write("KJR EVALUATION RESULT\n")
        for i, sentence in enumerate(matching_sentences, 1):
            sentence_text = sentence[0].replace('\n', '')  # Remove '\n' characters
            score = sentence[1]
            file.write(f"{i}. {sentence_text} [{score}]\n")
    
    cloudinary.config(
        cloud_name=os.getenv('cloud_name'),
        api_key=os.getenv('api_key'),
        api_secret=os.getenv('api_secret'),
        secure=True
    )
    
    result_cloudinary_url = upload(result_file, public_id = "result", folder="Idan", resource_type="auto")
    # Send the result file for download
    result = {"download_url" : result_cloudinary_url["secure_url"],
              "Message" : "Evaluation success"}
    
    return result

if __name__ == "__main__":
    app.run()