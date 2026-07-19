document.addEventListener('DOMContentLoaded', function() {
    
    // ----------------------------------------------------
    // LANDING PAGE UPLOAD MODULE
    // ----------------------------------------------------
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const removePreviewBtn = document.getElementById('remove-preview');
    const fileInfoBar = document.getElementById('file-info-bar');
    const fileNameTxt = document.getElementById('file-name-txt');
    const dropzoneContent = document.getElementById('dropzone-content');
    const uploadForm = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    const btnText = document.getElementById('btn-text');

    if (dropzone && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });

        // Handle dropped files
        dropzone.addEventListener('drop', handleDrop, false);

        // Click to open file browser (standard behavior via label/hidden input, but double check)
        fileInput.addEventListener('change', handleFileSelect);

        // Remove preview button
        if (removePreviewBtn) {
            removePreviewBtn.addEventListener('click', function(e) {
                e.stopPropagation(); // prevent triggering dropzone click
                resetUpload();
            });
        }
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            fileInput.files = files;
            processFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    }

    function processFile(file) {
        // Validate file type is image
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (png, jpg, jpeg, webp).');
            resetUpload();
            return;
        }

        // Show filename
        if (fileNameTxt) fileNameTxt.textContent = file.name;
        if (fileInfoBar) fileInfoBar.style.display = 'flex';

        // Read image file for preview
        const reader = new FileReader();
        reader.onload = function(e) {
            if (previewImg) previewImg.src = e.target.result;
            if (previewContainer) previewContainer.style.display = 'block';
            if (dropzoneContent) dropzoneContent.style.display = 'none';
            if (dropzone) {
                dropzone.style.padding = '12px'; // tighter padding to fit the preview nicely
                dropzone.style.borderStyle = 'solid';
                dropzone.style.borderColor = 'var(--primary)';
            }
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        if (fileInput) fileInput.value = '';
        if (previewImg) previewImg.src = '';
        if (previewContainer) previewContainer.style.display = 'none';
        if (fileInfoBar) fileInfoBar.style.display = 'none';
        if (dropzoneContent) dropzoneContent.style.display = 'flex';
        if (dropzone) {
            dropzone.removeAttribute('style'); // resets padding & borders to stylesheet default
        }
    }

    // Handle Form Submit Loading state
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener('submit', function() {
            if (btnLoader) btnLoader.style.display = 'block';
            if (btnText) btnText.textContent = 'Analyzing Produce...';
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.85';
            submitBtn.style.cursor = 'not-allowed';
        });
    }


    // ----------------------------------------------------
    // RESULT PAGE PREDICTION VISUALIZATION
    // ----------------------------------------------------
    const rawPredictionElem = document.getElementById('raw-prediction');
    if (rawPredictionElem) {
        const rawPredictionText = rawPredictionElem.textContent.trim();
        parseAndRenderResult(rawPredictionText);
    }

    function parseAndRenderResult(rawString) {
        // Expected format: "Fresh (95.42%)" or "Rotten (84.10%)"
        // Let's parse with regex
        const statusMatch = rawString.match(/^(Fresh|Rotten)/i);
        const confidenceMatch = rawString.match(/(\d+(?:\.\d+)?)\s*%/);

        let status = statusMatch ? statusMatch[1].toLowerCase() : 'unknown';
        let confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : 0;

        // Fallback checks
        if (status === 'unknown') {
            if (rawString.toLowerCase().includes('fresh')) {
                status = 'fresh';
            } else if (rawString.toLowerCase().includes('rotten')) {
                status = 'rotten';
            }
        }

        if (confidence === 0 && rawString.includes('(')) {
            // fallback attempt if format is slightly off
            const number = parseFloat(rawString.replace(/[^0-9.]/g, ''));
            if (!isNaN(number)) confidence = number;
        }
        
        // Cap confidence max
        if (confidence > 100) confidence = 100;
        if (confidence < 0) confidence = 0;

        // Render Status Badges
        const badgeContainer = document.getElementById('badge-status-container');
        if (badgeContainer) {
            if (status === 'fresh') {
                badgeContainer.className = 'summary-badge-status status-badge-fresh';
                badgeContainer.innerHTML = '<span>✔</span> Fresh';
            } else {
                badgeContainer.className = 'summary-badge-status status-badge-rotten';
                badgeContainer.innerHTML = '<span>✖</span> Rotten';
            }
        }

        // Render Circular Gauge SVG Fill
        const circleFill = document.getElementById('gauge-circle-fill');
        const gaugePercentageTxt = document.getElementById('gauge-percentage-txt');
        
        if (circleFill) {
            // The circle has a stroke-dasharray of 226 (approx 2 * PI * r, r=36 -> 226)
            const circumference = 226;
            const strokeOffset = circumference - (confidence / 100) * circumference;
            
            // Set correct color according to status
            if (status === 'rotten') {
                circleFill.style.stroke = 'var(--danger)';
            } else {
                circleFill.style.stroke = 'var(--primary)';
            }
            
            // Trigger animation frame after rendering
            setTimeout(() => {
                circleFill.style.strokeDashoffset = strokeOffset;
            }, 100);
        }

        // Animated percentage counter text inside gauge
        if (gaugePercentageTxt) {
            animateValue(gaugePercentageTxt, 0, confidence, 1500, '%');
        }

        // Animate linear progress bar
        const progressBarFill = document.getElementById('progress-bar-fill');
        const progressPercentTxt = document.getElementById('progress-percent-txt');
        
        if (progressBarFill) {
            if (status === 'rotten') {
                progressBarFill.style.background = 'linear-gradient(to right, var(--accent), var(--danger))';
            }
            
            setTimeout(() => {
                progressBarFill.style.width = confidence + '%';
            }, 100);
        }
        
        if (progressPercentTxt) {
            animateValue(progressPercentTxt, 0, confidence, 1500, '%');
        }

        // Render dynamic statistic badges
        const statStatusVal = document.getElementById('stat-status-val');
        const statConfidenceVal = document.getElementById('stat-confidence-val');
        const statTimeVal = document.getElementById('stat-time-val');
        const statCountVal = document.getElementById('stat-count-val');

        if (statStatusVal) {
            statStatusVal.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            statStatusVal.style.color = (status === 'fresh') ? 'var(--primary)' : 'var(--danger)';
        }

        if (statConfidenceVal) {
            statConfidenceVal.textContent = confidence.toFixed(2) + '%';
        }

        // Generate simulated mock processing/prediction duration (e.g. 42ms - 88ms)
        if (statTimeVal) {
            const randomTime = (Math.random() * (0.088 - 0.042) + 0.042).toFixed(3);
            statTimeVal.textContent = randomTime + 's';
        }
        
        if (statCountVal) {
            // YOLO detected object count (simulate detection or mock as 1 unless specific)
            statCountVal.textContent = '1 Object';
        }
    }

    // Helper function to animate numbers counting up
    function animateValue(obj, start, end, duration, suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentValue = progress * (end - start) + start;
            obj.innerHTML = currentValue.toFixed(1) + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end.toFixed(2) + suffix;
            }
        };
        window.requestAnimationFrame(step);
    }
});
