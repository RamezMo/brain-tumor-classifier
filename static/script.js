document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');

    // Show a preview of the uploaded image
    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                resultText.textContent = ''; // Clear previous result
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission

        const file = imageUpload.files[0];
        if (!file) {
            resultText.textContent = "Please select an image file.";
            return;
        }

        // Show loader and clear previous result
        loader.classList.remove('hidden');
        resultText.textContent = '';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display the result
            if (data.error) {
                resultText.textContent = `Error: ${data.error}`;
            } else {
                resultText.textContent = `${data.prediction} (${data.confidence})`;
            }

        } catch (error) {
            console.error('Error:', error);
            resultText.textContent = 'An error occurred. Please try again.';
        } finally {
            // Hide loader
            loader.classList.add('hidden');
        }
    });
});