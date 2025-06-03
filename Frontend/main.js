    const API_CONFIG = {
            baseURL: 'http://localhost:8000/api', // Change to your backend URL
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
// Add this function
function testChartWithMockData() {
    console.log('Testing chart with mock data...');
    
    const mockHypotheses = [
        { 
            complexity: 20, 
            informationDensity: 85, 
            text: "Low complexity leads to high information density in neural networks", 
            citationCount: 5, 
            validationStatus: "Highly Supported", 
            citations: ["Smith et al. 2023", "Jones 2024"] 
        },
        { 
            complexity: 45, 
            informationDensity: 70, 
            text: "Moderate complexity maintains good information transfer", 
            citationCount: 3, 
            validationStatus: "Partially Supported", 
            citations: ["Brown 2023"] 
        },
        { 
            complexity: 80, 
            informationDensity: 30, 
            text: "High complexity reduces information density due to noise", 
            citationCount: 1, 
            validationStatus: "Limited Support", 
            citations: ["Wilson 2024"] 
        }
    ];

    mockHypotheses.forEach((hypothesis, index) => {
        setTimeout(() => {
            console.log(`Adding data point ${index + 1}:`, hypothesis);
            displayHypothesis(hypothesis);
            updateChart(hypothesis);
        }, index * 2000); // 2 second intervals
    });
}


        // Global state management
        const state = {
            complexity: 50,
            currentHypothesis: null,
            chartData: [],
            percolationPoint: 75,
            isLoading: false,
            uploadedFiles: [],
            topic: '',
            domain: 'general',
            chart: null
        };

        // DOM elements cache
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

            // Check if all required elements exist
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
            
    setTimeout(() => {
        if (state.chart) {
            console.log('Auto-testing chart...');
            testChartWithMockData();
        }
    }, 2000);


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
                    },
                    {
                        label: 'Percolation Point',
                        data: [{ x: 75, y: 0 }, { x: 75, y: 100 }],
                        borderColor: 'rgba(239, 68, 68, 1)',  // Remove transparency - make it solid red
                        borderWidth: 4,                        // Make it thicker
                        borderDash: [8, 4],                   // Bigger dash pattern
                        pointRadius: 0,
                        fill: false,
                        tension: 0                            // Add this for straight line
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

       // Event listeners
        function bindEventListeners() {
            elements.complexitySlider.addEventListener('input', (e) => {
                state.complexity = parseInt(e.target.value);
                updateComplexityDisplay();
                updatePercolationIndicator();
                console.log('Complexity changed to:', state.complexity);
            });

            // Generate button
            elements.generateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Generate button clicked');
                generateHypothesis();
            });

            // Topic and domain inputs
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

            // File upload
            elements.literatureFile.addEventListener('change', handleFileUpload);
    const testBtn = document.getElementById('testChartBtn');
    if (testBtn) {
        testBtn.addEventListener('click', testChartWithMockData);
    }
        }

        // Update complexity display
        function updateComplexityDisplay() {
            elements.complexityValue.textContent = state.complexity;
            
            let label, color;
            if (state.complexity <= 30) {
                label = 'Simple';
                color = 'var(--accent-green)';
            } else if (state.complexity <= 70) {
                label = 'Moderate';
                color = 'var(--accent-yellow)';
            } else {
                label = 'Complex';
                color = 'var(--accent-red)';
            }
            
            elements.complexityLabel.textContent = label;
            elements.complexityBadge.style.color = color;
        }

        // Update percolation indicator
        function updatePercolationIndicator() {
            const indicator = elements.percolationIndicator;
            
            if (state.complexity <= 60) {
                indicator.className = 'percolation-indicator safe';
                indicator.innerHTML = '<strong>✅ Status:</strong> Safe Zone - High Information Density Expected';
            } else if (state.complexity <= 80) {
                indicator.className = 'percolation-indicator warning';
                indicator.innerHTML = '<strong>⚠️ Status:</strong> Approaching Percolation Point - Density May Decline';
            } else {
                indicator.className = 'percolation-indicator danger';
                indicator.innerHTML = '<strong>❌ Status:</strong> Beyond Percolation Point - Low Information Density';
            }
        }

        async function handleFileUpload(e) {
            const files = Array.from(e.target.files);

            for (const file of files) {
                if (!state.uploadedFiles.find(f => f.name === file.name)) {
                    try {
                        // Show upload progress
                        elements.fileUploadContainer.classList.add('uploading');

                        const formData = new FormData();
                        formData.append('file', file);
                        formData.append('domain', state.domain);

                        const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.uploadFile}`, {
                            method: 'POST',
                            body: formData
                        });

                        if (response.ok) {
                            const result = await response.json();
                            state.uploadedFiles.push({
                                name: file.name,
                                size: file.size,
                                id: result.fileId // Backend should return file ID
                            });
                            addFileToDisplay(file);
                        }
                    } catch (error) {
                        console.error('Upload failed:', error);
                        alert(`Failed to upload ${file.name}`);
                    } finally {
                        elements.fileUploadContainer.classList.remove('uploading');
                    }
                }
            }

            updateFileUploadDisplay();
        }

        // Add file to display
        function addFileToDisplay(file) {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'uploaded-file';
            fileDiv.innerHTML = `
                <span>📄 ${file.name} (${formatFileSize(file.size)})</span>
                <button class="remove-file" onclick="removeFile('${file.name}')">&times;</button>
            `;
            elements.uploadedFiles.appendChild(fileDiv);
        }

        // Remove file
        function removeFile(fileName) {
            state.uploadedFiles = state.uploadedFiles.filter(f => f.name !== fileName);
            updateFileUploadDisplay();
        }

        // Update file upload display
        function updateFileUploadDisplay() {
            elements.uploadedFiles.innerHTML = '';
            state.uploadedFiles.forEach(file => addFileToDisplay(file));
        }

        // Format file size
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // Generate hypothesis
        async function generateHypothesis() {
            if (state.isLoading) return;
            
            const topic = elements.topicInput.value.trim();
            if (!topic) {
                alert('Please enter a scientific topic or phenomenon.');
                return;
            }
            if(!uploadFile){
                alert('Please upload a scientific literture (pdf)');
                return;
            }
            setLoadingState(true);
            
           try {
                const formData = new FormData();
                 formData.append('phenomenon', topic);
                 formData.append('complexity', state.complexity); // assuming it's an int
                 formData.append('file_media', uploadFile);
                
                const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.generateHypothesis}`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const hypothesis = await response.json();
                
                displayHypothesis(hypothesis);
                updateChart(hypothesis);

            } catch (error) {
                console.error('Error generating hypothesis:', error);
                elements.hypothesisText.textContent = 'Error generating hypothesis. Please try again.';
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


       

        function displayHypothesis(hypothesis) {
            state.currentHypothesis = hypothesis;
            
            elements.hypothesisText.textContent = hypothesis.text;
            
            elements.densityValue.textContent = `${hypothesis.info_density.overall_quality}%``;
            elements.complexityScore.textContent = hypothesis.Complexity_Score;
            elements.validationStatus.textContent = hypothesis.validationStatus;
            elements.citationCount.textContent = hypothesis.citationCount;
            
            elements.hypothesisMetadata.style.display = 'grid';
            elements.citationsSection.style.display = hypothesis.citationCount > 0 ? 'block' : 'none';
            
            elements.citationsList.innerHTML = '';
            hypothesis.citations.forEach(citation => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation-item';
                citationDiv.textContent = citation;
                elements.citationsList.appendChild(citationDiv);
            });
            
            styleMetadataValues(hypothesis);
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

        function styleMetadataValues(hypothesis) {
            const density = hypothesis.informationDensity;
            if (density >= 60) {
                elements.densityValue.style.color = 'var(--accent-green)';
            } else if (density >= 30) {
                elements.densityValue.style.color = 'var(--accent-yellow)';
            } else {
                elements.densityValue.style.color = 'var(--accent-red)';
            }
            

            if (hypothesis.validationStatus === 'Highly Supported') {
                elements.validationStatus.style.color = 'var(--accent-green)';
            } else if (hypothesis.validationStatus === 'Partially Supported') {
                elements.validationStatus.style.color = 'var(--accent-yellow)';
            } else {
                elements.validationStatus.style.color = 'var(--accent-red)';
            }
        }

        // Update chart with new data point
        function updateChart(hypothesis) {
    if (!state.chart) {
        console.error('Chart not initialized');
        return;
    }
    
    state.chartData.push({
        x: hypothesis.Complexity_Score,
        y: hypothesis.info_density.overall_quality

    });
    

    if (state.chartData.length > 20) {
        state.chartData = state.chartData.slice(-20);
    }
    
    try {
        state.chart.data.datasets[1].data = [...state.chartData];
        state.chart.update();
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