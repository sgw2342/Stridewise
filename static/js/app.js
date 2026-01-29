// Injury Prediction App - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Data source selection
    const connectGarminBtn = document.getElementById('connect-garmin-btn');
    const createSampleBtn = document.getElementById('create-sample-btn');
    const garminDialog = document.getElementById('garmin-dialog');
    const closeGarminDialog = document.getElementById('close-garmin-dialog');
    const closeGarminBtn = document.getElementById('close-garmin-btn');
    const dataSourceSection = document.getElementById('data-source-section');
    const predictionForm = document.getElementById('prediction-form');
    
    // Garmin connection dialog
    if (connectGarminBtn) {
        connectGarminBtn.addEventListener('click', function() {
            garminDialog.classList.remove('hidden');
        });
    }
    
    if (closeGarminDialog) {
        closeGarminDialog.addEventListener('click', function() {
            garminDialog.classList.add('hidden');
        });
    }
    
    if (closeGarminBtn) {
        closeGarminBtn.addEventListener('click', function() {
            garminDialog.classList.add('hidden');
        });
    }
    
    // Close dialog when clicking outside
    if (garminDialog) {
        garminDialog.addEventListener('click', function(e) {
            if (e.target === garminDialog) {
                garminDialog.classList.add('hidden');
            }
        });
    }
    
    // Create sample data button
    if (createSampleBtn) {
        createSampleBtn.addEventListener('click', function() {
            window.location.href = '/create_data';
        });
    }
    
    // Always show data source selection first by default
    // Only show prediction form if URL explicitly has ?show_form=true parameter
    // This ensures start screen is always shown on initial load
    
    // Check URL for parameter indicating form should be shown
    const urlParams = new URLSearchParams(window.location.search);
    const showForm = urlParams.get('show_form');
    
    // Only show prediction form if explicitly requested via URL parameter
    if (showForm === 'true' && dataSourceSection && predictionForm) {
        dataSourceSection.classList.add('hidden');
        predictionForm.classList.remove('hidden');
    }
    // Otherwise, data source selection stays visible (default state)
    
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const backToDataBtn = document.getElementById('back-to-data-btn');
    
    // Back to data creation button
    if (backToDataBtn) {
        backToDataBtn.addEventListener('click', function() {
            window.location.href = '/create_data';
        });
    }
    
    if (!form) return; // Form might not exist on create_data page
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results
        resultContainer.classList.add('hidden');
        error.classList.add('hidden');
        loading.classList.remove('hidden');
        
        // Get form data
        const illness = document.getElementById('illness').checked;
        const pains = document.getElementById('pains').checked;
        
        try {
            // Make API request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    illness: illness,
                    pains: pains
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'An error occurred');
            }
            
            // Display result
            displayResult(data);
            
        } catch (err) {
            displayError(err.message);
        } finally {
            loading.classList.add('hidden');
        }
    });
    
    function displayResult(data) {
        const icon = document.getElementById('result-icon');
        const message = document.getElementById('result-message');
        const details = document.getElementById('result-details');
        
        // Set icon based on risk level
        let iconSymbol, containerClass;
        switch(data.risk_level) {
            case 'green':
                iconSymbol = '✅';
                containerClass = 'green';
                break;
            case 'orange':
                iconSymbol = '⚠️';
                containerClass = 'orange';
                break;
            case 'red':
                iconSymbol = '🔴';
                containerClass = 'red';
                break;
            default:
                iconSymbol = '❓';
                containerClass = 'green';
        }
        
        icon.textContent = iconSymbol;
        message.textContent = `${getRiskLevelText(data.risk_level)} (${data.risk_score}% risk)`;
        details.textContent = data.message;
        
        // Set container class
        resultContainer.className = `result-container ${containerClass}`;
        
        // Show result
        resultContainer.classList.remove('hidden');
        
        // Scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    function getRiskLevelText(level) {
        switch(level) {
            case 'green':
                return 'All Clear - Good to Train';
            case 'orange':
                return 'Caution - Modify Training';
            case 'red':
                return 'High Risk - Consider Rest';
            default:
                return 'Assessment Complete';
        }
    }
    
    function displayError(errorMessage) {
        error.textContent = `Error: ${errorMessage}`;
        error.classList.remove('hidden');
    }
});
