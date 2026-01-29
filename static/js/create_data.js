// Create Data Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('data-config-form');
    const configFormContainer = document.getElementById('config-form-container');
    const configSaved = document.getElementById('config-saved');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    
    const editConfigBtn = document.getElementById('edit-config-btn');
    const returnHomeBtn = document.getElementById('return-home-btn');
    
    // Checkboxes for conditional sections
    const includeSprint = document.getElementById('include-sprint');
    const includeLong = document.getElementById('include-long');
    const includeTempo = document.getElementById('include-tempo');
    
    const sprintDetails = document.getElementById('sprint-details');
    const longDetails = document.getElementById('long-details');
    const tempoDetails = document.getElementById('tempo-details');
    
    // Toggle conditional sections
    if (includeSprint) {
        includeSprint.addEventListener('change', function() {
            if (this.checked) {
                sprintDetails.classList.remove('hidden');
            } else {
                sprintDetails.classList.add('hidden');
            }
        });
    }
    
    if (includeLong) {
        includeLong.addEventListener('change', function() {
            if (this.checked) {
                longDetails.classList.remove('hidden');
            } else {
                longDetails.classList.add('hidden');
            }
        });
    }
    
    if (includeTempo) {
        includeTempo.addEventListener('change', function() {
            if (this.checked) {
                tempoDetails.classList.remove('hidden');
            } else {
                tempoDetails.classList.add('hidden');
            }
        });
    }
    
    // Load saved configuration if it exists
    loadSavedConfig();
    
    // Form submission
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Hide error
            error.classList.add('hidden');
            loading.classList.remove('hidden');
            
            // Collect form data
            const formData = {
                include_sprint: includeSprint.checked,
                include_long: includeLong.checked,
                include_tempo: includeTempo.checked,
                sprint_sessions: includeSprint.checked ? parseInt(document.getElementById('sprint-sessions').value) : 0,
                sprint_kms: includeSprint.checked ? parseFloat(document.getElementById('sprint-kms').value) : 0,
                long_sessions: includeLong.checked ? parseInt(document.getElementById('long-sessions').value) : 0,
                long_distance: includeLong.checked ? parseFloat(document.getElementById('long-distance').value) : 0,
                tempo_sessions: includeTempo.checked ? parseInt(document.getElementById('tempo-sessions').value) : 0,
                tempo_zone: includeTempo.checked ? parseInt(document.getElementById('tempo-zone').value) : 0,
                tempo_kms: includeTempo.checked ? parseFloat(document.getElementById('tempo-kms').value) : 0,
                created_at: new Date().toISOString()
            };
            
            try {
                // Save configuration
                const response = await fetch('/save_config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to save configuration');
                }
                
                // Show success state
                configFormContainer.classList.add('hidden');
                configSaved.classList.remove('hidden');
                loading.classList.add('hidden');
                
                // Store in localStorage for persistence
                localStorage.setItem('user_data_config', JSON.stringify(formData));
                localStorage.setItem('user_data_configured', 'true');
                
            } catch (err) {
                loading.classList.add('hidden');
                error.textContent = `Error: ${err.message}`;
                error.classList.remove('hidden');
            }
        });
    }
    
    // Edit configuration button
    if (editConfigBtn) {
        editConfigBtn.addEventListener('click', function() {
            configSaved.classList.add('hidden');
            configFormContainer.classList.remove('hidden');
        });
    }
    
    // Regenerate data button
    const regenerateBtn = document.getElementById('regenerate-data-btn');
    if (regenerateBtn) {
        regenerateBtn.addEventListener('click', async function() {
            // Show loading
            configSaved.classList.add('hidden');
            loading.classList.remove('hidden');
            error.classList.add('hidden');
            
            // Get current configuration
            const saved = localStorage.getItem('user_data_config');
            if (!saved) {
                error.textContent = 'Error: No configuration found. Please save your configuration first.';
                error.classList.remove('hidden');
                loading.classList.add('hidden');
                configSaved.classList.remove('hidden');
                return;
            }
            
            try {
                const config = JSON.parse(saved);
                
                // Call save_config endpoint which will regenerate data
                const response = await fetch('/save_config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(config)
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Failed to regenerate data');
                }
                
                // Show success
                loading.classList.add('hidden');
                configSaved.classList.remove('hidden');
                
                // Show success message
                const successMsg = document.querySelector('.success-message p:last-child');
                if (successMsg) {
                    successMsg.textContent = 'Data has been regenerated successfully with your current configuration.';
                }
                
            } catch (err) {
                loading.classList.add('hidden');
                configSaved.classList.remove('hidden');
                error.textContent = `Error: ${err.message}`;
                error.classList.remove('hidden');
            }
        });
    }
    
    // Return home button
    if (returnHomeBtn) {
        returnHomeBtn.addEventListener('click', function() {
            // Navigate to home with parameter to show prediction form
            window.location.href = '/?show_form=true';
        });
    }
    
    function loadSavedConfig() {
        const saved = localStorage.getItem('user_data_config');
        if (saved) {
            try {
                const config = JSON.parse(saved);
                
                // Restore form values
                if (config.include_sprint) {
                    includeSprint.checked = true;
                    sprintDetails.classList.remove('hidden');
                    document.getElementById('sprint-sessions').value = config.sprint_sessions || 0;
                    document.getElementById('sprint-kms').value = config.sprint_kms || 0;
                }
                
                if (config.include_long) {
                    includeLong.checked = true;
                    longDetails.classList.remove('hidden');
                    document.getElementById('long-sessions').value = config.long_sessions || 0;
                    document.getElementById('long-distance').value = config.long_distance || 0;
                }
                
                if (config.include_tempo) {
                    includeTempo.checked = true;
                    tempoDetails.classList.remove('hidden');
                    document.getElementById('tempo-sessions').value = config.tempo_sessions || 0;
                    document.getElementById('tempo-zone').value = config.tempo_zone || 3;
                    document.getElementById('tempo-kms').value = config.tempo_kms || 0;
                }
            } catch (e) {
                console.error('Error loading saved config:', e);
            }
        }
    }
});
