    const API_CONFIG = {
            baseURL: 'https://percolation-hypotheses-gen.onrender.com', // Change to your backend URL
            endpoints: {
                generateHypothesis: '/generate-hypothesis',
                uploadFile: '/upload-literature',
                validateHypothesis: '/validate-hypothesis'
            },
            timeout: 30000
        };
        async function apiCall(endpoint, options = {}) {
            try {
                const response = await fetch(`${API_CONFIG.baseURL}${endpoint}`, {
                    timeout: API_CONFIG.timeout,
                    ...options
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.message || `HTTP ${response.status}`);
                }

                return await response.json();
            } catch (error) {
                console.error('API call failed:', error);
                throw error;
            }
        }

        // Global state management
        const state = {
            complexity: 5,
            currentHypothesis: null,
            chartData: [],
            percolationPoint: null,
            isLoading: false,
            uploadedFiles: [],
            topic: '',
            domain: 'general',
            chart: null,
            uploadedFile: null,
            percolationDetected: false,
            percolationThreshold: null,
            dataPoints: [],
        };

     
        const elements = {
            complexitySlider: document.getElementById('complexitySlider'),
            complexityValue: document.getElementById('complexityValue'),
            complexityBadge: document.getElementById('complexityBadge'),
            complexityLabel: document.getElementById('complexityLabel'),
            generateBtn: document.getElementById('generateBtn'),
            generateBtnText: document.getElementById('generateBtnText'),
            percolationIndicator: document.getElementById('percolationIndicator'),
            hypothesisText: document.getElementById('hypothesisText'),
            hypothesisMetadata: document.getElementById('hypothesisMetadata'),
            citationsSection: document.getElementById('citationsSection'),
            loadingSpinner: document.getElementById('loadingSpinner'),
            timestamp: document.getElementById('timestamp'),
            statusText: document.getElementById('statusText'),
            topicInput: document.getElementById('topicInput'),
            domainSelect: document.getElementById('domain'),
            literatureFile: document.getElementById('literatureFile'),
            uploadedFiles: document.getElementById('uploadedFiles'),
            fileUploadContainer: document.getElementById('fileUploadContainer'),
            densityValue: document.getElementById('densityValue'),
            complexityScore: document.getElementById('complexityScore'),
            validationStatus: document.getElementById('validationStatus'),
            citationCount: document.getElementById('citationCount'),
            citationsList: document.getElementById('citationsList')
        };

       function initializeApp() {
            console.log('Initializing app...');

            const requiredElements = ['complexitySlider', 'generateBtn', 'densityChart'];
            for (const elementId of requiredElements) {
                const element = document.getElementById(elementId);
                if (!element) {
                    console.error(`Required element not found: ${elementId}`);
                    return;
                }
            }

            initializeChart();
            bindEventListeners();
            updateUI();
            updateTimestamp();

            console.log('App initialized successfully');   

            }
    function initializeChart() {

        
    const canvas = document.getElementById('densityChart');
    if (!canvas) {
        console.error('Chart canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
   if (typeof Chart === 'undefined') {
    console.error('Chart.js not loaded');
    document.getElementById('densityChart').parentElement.innerHTML = 
        '<p style="text-align:center;color:#a0a0a0;">Chart loading...</p>';
    return;


}
    try {
        state.chart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Generated Hypotheses',
                        data: state.chartData,
                        borderColor: 'rgba(34, 197, 94, 1)',
                        backgroundColor: 'rgba(34, 197, 94, 0.8)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBackgroundColor: 'rgba(34, 197, 94, 1)',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#e0e0e0',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#e0e0e0',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,            
                        max: 100,
                        title: {
                            display: true,
                            text: 'Complexity Level',
                            color: '#e0e0e0',
                            font: { size: 14, weight: 'bold' }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#a0a0a0'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Information Density',
                            color: '#e0e0e0',
                            font: { size: 14, weight: 'bold' }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#a0a0a0'
                        },
                        min: 0,
                        max: 100
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
        console.log('Chart initialized successfully');
    } catch (error) {
        console.error('Error initializing chart:', error);
    }
}
function updatePercolationLine(percolationPoint) {
    if (!state.chart) return;
    
    // Update the percolation line position
    state.chart.data.datasets[1].data = [
        { x: percolationPoint, y: 0 }, 
        { x: percolationPoint, y: 100 }
    ];
    
    // Make it visible
    state.chart.data.datasets[1].hidden = false;
    state.chart.update();
}
    function   detectPercolationPoint() {
    if (state.dataPoints.length < 5) return; 
 
    const sortedData = [...state.dataPoints].sort((a, b) => a.complexity - b.complexity);
    
    let maxDensity = 0;
    let peakComplexity = 0;
    let significantDrop = false;
    let percolationPoint = null;
    for (let i = 0; i < sortedData.length; i++) {
        if (sortedData[i].density > maxDensity) {
            maxDensity = sortedData[i].density;
            peakComplexity = sortedData[i].complexity;
        }
    }
    
    for (let i = 0; i < sortedData.length; i++) {
        if (sortedData[i].complexity > peakComplexity) {
            const densityDrop = maxDensity - sortedData[i].density;
            const dropPercentage = (densityDrop / maxDensity) * 100;
            
            if (dropPercentage >= 20) { 
                percolationPoint = sortedData[i].complexity;
                significantDrop = true;
                break;
            }
        }
    }
    
    if (significantDrop && percolationPoint) {
        state.percolationDetected = true;
        state.percolationThreshold = percolationPoint;
        updatePercolationLine(percolationPoint);
        console.log(`Percolation point detected at complexity: ${percolationPoint}`);
    }
}
        function bindEventListeners() {
            elements.complexitySlider.addEventListener('input', (e) => {
                state.complexity = parseInt(e.target.value);
                updateComplexityDisplay();
                updatePercolationIndicator();
                console.log('Complexity changed to:', state.complexity);
            });

   
            elements.generateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Generate button clicked');
                generateHypothesis();
            });

            elements.topicInput.addEventListener('input', (e) => {
                state.topic = e.target.value;
                updateStatus();
                console.log('Topic changed to:', state.topic);
            });

            elements.domainSelect.addEventListener('change', (e) => {
                state.domain = e.target.value;
                updateStatus();
                console.log('Domain changed to:', state.domain);
            });

     
            elements.literatureFile.addEventListener('change', handleFileUpload);
    
        }
      
        function updateComplexityDisplay() {
            elements.complexityValue.textContent = state.complexity;
            
            let label, color;
            if (state.complexity <= 3) {
                label = 'Simple';
                color = 'var(--accent-green)';
            } else if (state.complexity <= 7) {
                label = 'Moderate';
                color = 'var(--accent-yellow)';
            } else {
                label = 'Complex';
                color = 'var(--accent-red)';
            }
            
            elements.complexityLabel.textContent = label;
            elements.complexityBadge.style.color = color;
        }

        function updatePercolationIndicator() {
            const indicator = elements.percolationIndicator;
            
            if (state.percolationDetected) {
        if (state.complexity >= state.percolationThreshold) {
            indicator.className = 'percolation-indicator danger';
            indicator.innerHTML = `<strong>❌ Status:</strong> Beyond Detected Percolation Point (${state.percolationThreshold.toFixed(1)}) - Low Information Density`;
        } else {
            indicator.className = 'percolation-indicator warning';
            indicator.innerHTML = `<strong>⚠️ Status:</strong> Approaching Detected Percolation Point (${state.percolationThreshold.toFixed(1)})`;
        }
    } else {
        // Original logic when no percolation detected yet
        if (state.complexity <= 6) {
            indicator.className = 'percolation-indicator safe';
            indicator.innerHTML = '<strong>✅ Status:</strong> Safe Zone - Collecting data to detect percolation point';
        } else if (state.complexity <= 8) {
            indicator.className = 'percolation-indicator warning';
            indicator.innerHTML = '<strong>⚠️ Status:</strong> High complexity - Monitor for percolation effects';
        } else {
            indicator.className = 'percolation-indicator danger';
            indicator.innerHTML = '<strong>❌ Status:</strong> Very high complexity - Potential percolation risk';
        }
    }
        }

        async function handleFileUpload(e) {
    const files = Array.from(e.target.files);
    
    if (files.length > 0) {
        const file = files[0]; 
        state.uploadedFile = file; // Store the actual file object
        
        // Update display without uploading to server yet
        state.uploadedFiles = [{ name: file.name, size: file.size }];
        updateFileUploadDisplay();
        console.log('File selected:', file.name);
    }
}

        function addFileToDisplay(file) {
    const fileDiv = document.createElement('div');
    fileDiv.className = 'uploaded-file';
    fileDiv.innerHTML = `
        <span>📄 ${file.name} (${formatFileSize(file.size)})</span>
        <button class="remove-file" onclick="removeFile('${file.name.replace(/'/g, "\\'")}')">×</button>
    `;
    elements.uploadedFiles.appendChild(fileDiv);
}

 
       function removeFile(fileName) {
    state.uploadedFiles = state.uploadedFiles.filter(f => f.name !== fileName);
    state.uploadedFile = null; // CLEAR THE ACTUAL FILE TOO
    updateFileUploadDisplay();
    
    // Also clear the file input
    if (elements.literatureFile) {
        elements.literatureFile.value = '';
    }
}

        function updateFileUploadDisplay() {
            elements.uploadedFiles.innerHTML = '';
            state.uploadedFiles.forEach(file => addFileToDisplay(file));
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }


        async function generateHypothesis() {
            if (state.isLoading) return;
 
    const topic = elements.topicInput.value.trim();
    if (!topic) {
        alert('Please enter a scientific topic or phenomenon.');
        elements.topicInput.focus();
        return;
    }
    
    if (!state.uploadedFile) {
        alert('Please upload a scientific literature (PDF file).');
        elements.literatureFile.focus();
        return;
    }
    
    if (state.uploadedFile.size > 10 * 1024 * 1024) { // 10MB limit
        alert('File size must be less than 10MB.');
        return;
    }
    
    if (!state.uploadedFile.name.toLowerCase().endsWith('.pdf')) {
        alert('Please upload a PDF file.');
        return;
    }
    

            setLoadingState(true);

            try {
                const formData = new FormData();
                formData.append('phenomenon', topic);
                formData.append('complexity', state.complexity.toString());
                formData.append('file_media', state.uploadedFile); // CHANGE: Use correct variable

                 console.log('Sending request with:', {
            phenomenon: topic,
            complexity: state.complexity,
            fileName: state.uploadedFile.name,
            fileSize: state.uploadedFile.size
        });
        

                const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.generateHypothesis}`, {
                    method: 'POST',
                    body: formData
                });
             console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
               if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
                console.error('Error response:', errorData);
            } catch (e) {
                console.error('Could not parse error response');
            }
            throw new Error(errorMessage);
        }
            
                const result = await response.json(); 
                console.log('Success response:', result);
                console.log('Complexity_Score value:', result.Complexity_Score);
        console.log('Complexity_Score type:', typeof result.Complexity_Score);
        console.log('Is it exactly 0?', result.Complexity_Score === 0);
                displayHypothesis(result); 
                updateChart(result);
            
            } catch (error) {
                console.error('Error generating hypothesis:', error);
        

        let userMessage = 'Error generating hypothesis. ';
        if (error.message.includes('Failed to fetch')) {
            userMessage += 'Please check your internet connection and try again.';
        } else if (error.message.includes('timeout')) {
            userMessage += 'Request timed out. Please try again.';
        } else {
            userMessage += error.message;
        }
        
        alert(userMessage);
        elements.hypothesisText.textContent = userMessage;
            } finally {
                setLoadingState(false);
            }
            }       

        function setLoadingState(loading) {
        state.isLoading = loading;
        elements.generateBtn.disabled = loading;
        elements.generateBtn.classList.toggle('loading', loading);
        elements.loadingSpinner.style.display = loading ? 'flex' : 'none';
        elements.statusText.textContent = loading ? 'Generating...' : 'Ready';
            }


       

        function displayHypothesis(result) {    
              if (!result) {
        console.error('No result provided to displayHypothesis');
        return;
    }         
    console.log('Displaying hypothesis with result:', result);
            state.currentHypothesis = result;
            
            elements.hypothesisText.textContent = result.hypothesis || 'No hypothesis generated';
            const info_density = result.info_density?.overall_quality ?? 0;
             elements.densityValue.textContent = `${(info_density *100).toFixed(3)}%`;
    elements.complexityScore.textContent = (result.Complexity_Score ?? 0).toFixed(1);
    elements.validationStatus.textContent = result.validationStatus || 'Pending';
       const citations = result.citations || [];
    elements.citationCount.textContent = citations.length;
    
            
            elements.hypothesisMetadata.style.display = 'grid';
            elements.citationsSection.style.display = result.citationCount > 0 ? 'block' : 'none';
            
            elements.citationsList.innerHTML = '';
               if (citations.length > 0) {
        elements.citationsSection.style.display = 'block';
        citations.forEach(citation => {
            const citationDiv = document.createElement('div');
            citationDiv.className = 'citation-item';
            citationDiv.textContent = citation;
            elements.citationsList.appendChild(citationDiv);
        });
    }
            styleMetadataValues(result);
        }

        async function validateHypothesis(hypothesis) {
            try {
                const validation = await apiCall(API_CONFIG.endpoints.validateHypothesis, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: hypothesis.text,
                        domain: state.domain 
                    })
                });

                elements.validationStatus.textContent = validation.status;
                styleMetadataValues({...hypothesis, validationStatus: validation.status});
            } catch (error) {
                console.error('Validation failed:', error);
            }
            }

        function styleMetadataValues(result) {
            const density = (result.info_density?.overall_quality?? 0 ) * 100;
            if (density >= 60) {
                elements.densityValue.style.color = 'var(--accent-green)';
            } else if (density >= 30) {
                elements.densityValue.style.color = 'var(--accent-yellow)';
            } else {
                elements.densityValue.style.color = 'var(--accent-red)';
            }
            
             const validationStatus = result.validationStatus || 'Pending';
    if (validationStatus === 'Highly Supported') {
        elements.validationStatus.style.color = 'var(--accent-green)';
    } else if (validationStatus === 'Partially Supported') {
        elements.validationStatus.style.color = 'var(--accent-yellow)';
    } else {
        elements.validationStatus.style.color = 'var(--accent-red)';
    }
}

       
        function updateChart(result) {
    if (!state.chart) {
        console.error('Chart not initialized');
        return;
    }
    const complexityScore = result.Complexity_Score ?? 0;
     const info_density = (result.info_density?.overall_quality ?? 0) * 100; 
    state.dataPoints.push({
        complexity: complexityScore,
        density: info_density
    });
    state.chartData.push({
        x: complexityScore,
        y: info_density

    });
    

    if (state.chartData.length > 20) {
        state.chartData = state.chartData.slice(-20);
         state.dataPoints = state.dataPoints.slice(-20);
    }
    
    try {
        state.chart.data.datasets[0].data = [...state.chartData];
        state.chart.update();
        detectPercolationPoint();
        console.log('Chart updated with new data point');
    } catch (error) {
        console.error('Error updating chart:', error);
    }
}

        function updateStatus() {
            if (!state.isLoading) {
                const hasRequirements = state.topic.length > 0;
                elements.statusText.textContent = hasRequirements ? 'Ready' : 'Enter topic';
            }
        }

        function updateTimestamp() {
            const now = new Date();
            elements.timestamp.textContent = now.toLocaleTimeString();
        }

        function updateUI() {
            updateComplexityDisplay();
            updatePercolationIndicator();
            updateStatus();
            setInterval(updateTimestamp, 1000);
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        document.addEventListener('DOMContentLoaded', initializeApp);
        
    