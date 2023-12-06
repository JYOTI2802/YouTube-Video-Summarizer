from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

app = Flask(__name__)

# Initialize the T5 model and tokenizer with model_max_length set
model_max_length = 1024  # Set your preferred max length
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=model_max_length)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def submit():
    try:
        youtube_url = request.args.get('youtube_url')
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is missing in query parameters'}), 400
        
         # Extract video ID from the YouTube URL using regex
        video_id_match = re.search(r'(?:v=|shorts/)([^&?/]+)', youtube_url)
        if video_id_match:
            video_id = video_id_match.group(1)
        else:
            return jsonify({'error': 'Invalid YouTube URL provided'}), 400

        # Fetch transcripts for the specified video ID
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        english_transcript = transcript_list.find_transcript(['en'])

        if english_transcript:
            transcript_segments = english_transcript.fetch()
            transcript_text = ' '.join([segment['text'] for segment in transcript_segments])

            # Generate summary using T5 model
            inputs = tokenizer.encode("summarize: " + transcript_text, return_tensors="pt", max_length=model_max_length, truncation=True)
            outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(outputs[0])
            # Removing <pad> and </s> tokens from the generated summary
            filtered_summary = re.sub(r'<pad>|</s>', '', summary)

            return render_template('summary.html', summary=filtered_summary)
            
        else:
            return jsonify({'message': 'No English transcript available'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
