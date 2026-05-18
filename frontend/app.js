// Configuration — always talk to whichever host is serving this page
const API_BASE_URL = window.location.origin;
let currentMode = 'search'; // 'search' or 'predict'
let map;
let markers = [];
let searchCircle = null;
let currentClickMarker = null;
let liveScrapedMarkerStacks = []; // Array of Arrays to match backend LIFO Stack

// DOM Elements
const btnModeSearch = document.getElementById('btn-mode-search');
const btnModePredict = document.getElementById('btn-mode-predict');
const btnDarkMode = document.getElementById('btn-dark-mode');
let isDarkMode = false;
const panelSearch = document.getElementById('panel-search');
const panelPredict = document.getElementById('panel-predict');
const radiusSlider = document.getElementById('radius-slider');
const radiusValue = document.getElementById('radius-value');
const searchResultsSummary = document.getElementById('search-results-summary');
const spinner = document.getElementById('loading-spinner');

// Predict Form Elements
const predictForm = document.getElementById('predict-form');
const inputLat = document.getElementById('pred-lat');
const inputLng = document.getElementById('pred-lng');
const btnPredict = document.getElementById('btn-predict');
const predictionResultBox = document.getElementById('prediction-result');
const priceValueText = document.getElementById('price-value');

// Scraper Admin Elements
const btnTriggerScrape = document.getElementById('btn-trigger-scrape');
const btnProcessQueue = document.getElementById('btn-process-queue');
const btnUndoScrape = document.getElementById('btn-undo-scrape');
const btnClearScraped = document.getElementById('btn-clear-scraped');
const scraperLog = document.getElementById('scraper-log');

// Formatters
const currencyFormatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
});

// Initialize App
function init() {
    // Center map on King County (Seattle area)
    // Center map on King County (Seattle area), move zoom to bottom right
    map = L.map('map', {zoomControl: false}).setView([47.6062, -122.3321], 10);
    L.control.zoom({position: 'bottomright'}).addTo(map);

    // Add CartoDB Positron TileLayer (clean and modern)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // Event Listeners
    setupEventListeners();
}

function setupEventListeners() {
    // Mode Switching
    btnModeSearch.addEventListener('click', () => switchMode('search'));
    btnModePredict.addEventListener('click', () => switchMode('predict'));

    // Radius Slider
    radiusSlider.addEventListener('input', (e) => {
        radiusValue.textContent = e.target.value;
        if (currentMode === 'search' && searchCircle) {
            searchCircle.setRadius(e.target.value * 1000); // meters
        }
    });

    // Map Click
    map.on('click', handleMapClick);

    // Predict Form Submit
    predictForm.addEventListener('submit', handlePredictionSubmit);

    // Scraper Controls
    btnTriggerScrape.addEventListener('click', handleTriggerScrape);
    btnProcessQueue.addEventListener('click', handleProcessQueue);
    btnUndoScrape.addEventListener('click', handleUndoScrape);
    btnClearScraped.addEventListener('click', handleClearScraped);
    btnDarkMode.addEventListener('click', toggleDarkMode);
}

function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode', isDarkMode);
    btnDarkMode.textContent = isDarkMode ? '☀️' : '🌙';
    
    // Switch tile layer
    const tileUrl = isDarkMode 
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';
    
    // Remove old tile layers
    map.eachLayer(layer => {
        if (layer instanceof L.TileLayer) {
            map.removeLayer(layer);
        }
    });

    L.tileLayer(tileUrl, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);
    
    // Ensure tile layer stays at the bottom
    map.eachLayer(layer => {
        if (layer instanceof L.TileLayer) {
            layer.bringToBack();
        }
    });
}

function switchMode(mode) {
    currentMode = mode;
    
    // Update Tabs
    btnModeSearch.classList.toggle('active', mode === 'search');
    btnModePredict.classList.toggle('active', mode === 'predict');
    
    // Update Panels
    panelSearch.classList.toggle('active', mode === 'search');
    panelPredict.classList.toggle('active', mode === 'predict');

    // Clean up map
    clearMap();
    if (mode === 'predict') {
        predictionResultBox.classList.add('hidden');
        inputLat.value = '';
        inputLng.value = '';
        btnPredict.disabled = true;
    } else {
        searchResultsSummary.innerHTML = 'Click map to search.';
    }
}

function clearMap() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    if (searchCircle) {
        map.removeLayer(searchCircle);
        searchCircle = null;
    }
    if (currentClickMarker) {
        map.removeLayer(currentClickMarker);
        currentClickMarker = null;
    }
}

async function handleMapClick(e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;

    clearMap();

    // Drop a pin where user clicked
    currentClickMarker = L.marker([lat, lng]).addTo(map);

    if (currentMode === 'search') {
        const radiusKm = parseFloat(radiusSlider.value);
        
        // Draw Search Circle
        searchCircle = L.circle([lat, lng], {
            color: '#4F46E5',
            fillColor: '#4F46E5',
            fillOpacity: 0.1,
            radius: radiusKm * 1000
        }).addTo(map);

        await performSearch(lat, lng, radiusKm);
    } else if (currentMode === 'predict') {
        // Populate inputs
        inputLat.value = lat.toFixed(4);
        inputLng.value = lng.toFixed(4);
        btnPredict.disabled = false;
        predictionResultBox.classList.add('hidden');
    }
}

async function performSearch(lat, lng, radiusKm) {
    toggleSpinner(true);
    searchResultsSummary.innerHTML = 'Searching properties...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/search?lat=${lat}&lon=${lng}&radius_km=${radiusKm}&limit=100`);
        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        
        if (data.length === 0) {
            searchResultsSummary.innerHTML = 'No properties found in this area.';
            toggleSpinner(false);
            return;
        }

        if (data.length === 100) {
            searchResultsSummary.innerHTML = `Found <strong>>100</strong> properties within ${radiusKm}km.`;
        } else {
            searchResultsSummary.innerHTML = `Found <strong>${data.length}</strong> properties within ${radiusKm}km.`;
        }

        // Add markers
        data.forEach(prop => {
            const propLat = prop.lat;
            const propLng = prop.long;
            
            const marker = L.circleMarker([propLat, propLng], {
                radius: 6,
                fillColor: '#10b981',
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map);

            const popupContent = `
                <h4>Property Detail</h4>
                <p><strong>Value:</strong> ${currencyFormatter.format(prop.price)}</p>
                <p><strong>Bedrooms:</strong> ${prop.bedrooms}</p>
                <p><strong>Bathrooms:</strong> ${prop.bathrooms}</p>
                <p><strong>Sqft Living:</strong> ${prop.sqft_living}</p>
                <p><strong>Year Built:</strong> ${prop.yr_built}</p>
                <p><strong>Distance:</strong> ${prop.distance_km} km</p>
            `;
            marker.bindPopup(popupContent);
            markers.push(marker);
        });

    } catch (error) {
        console.error("Search failed:", error);
        searchResultsSummary.innerHTML = '<span style="color:red">Error fetching results. Ensure API is running.</span>';
    } finally {
        toggleSpinner(false);
    }
}

async function handlePredictionSubmit(e) {
    e.preventDefault();
    toggleSpinner(true);
    predictionResultBox.classList.add('hidden');

    const payload = {
        lat: parseFloat(inputLat.value),
        long: parseFloat(inputLng.value),
        bedrooms: parseFloat(document.getElementById('pred-beds').value),
        bathrooms: parseFloat(document.getElementById('pred-baths').value),
        sqft_living: parseFloat(document.getElementById('pred-sqft').value),
        yr_built: parseFloat(document.getElementById('pred-year').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error('Prediction API failed');
        
        const data = await response.json();
        
        // Show result
        priceValueText.textContent = currencyFormatter.format(data.predicted_price_usd);
        predictionResultBox.classList.remove('hidden');

        // Add a popup to the map marker
        if (currentClickMarker) {
            currentClickMarker.bindPopup(`
                <h4>Prediction</h4>
                <p style="font-size:1.2rem; font-weight:bold; color:#10b981;">
                    ${currencyFormatter.format(data.predicted_price_usd)}
                </p>
            `).openPopup();
        }

    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Failed to predict price. Make sure the API is running and model is trained.");
    } finally {
        toggleSpinner(false);
    }
}

function toggleSpinner(show) {
    spinner.classList.toggle('hidden', !show);
}

// ---------------------------------------------------------------------------
// Scraper Admin Logic
// ---------------------------------------------------------------------------

function updateScraperLog(message, isError = false) {
    scraperLog.innerHTML = `<span style="color: ${isError ? '#dc2626' : 'inherit'}">${message}</span>`;
}

async function handleTriggerScrape() {
    toggleSpinner(true);
    updateScraperLog('Triggering scraper...');
    try {
        const response = await fetch(`${API_BASE_URL}/scrape/trigger`);
        if (!response.ok) throw new Error('Failed to trigger scraper');
        const data = await response.json();
        updateScraperLog(`${data.message}<br>Queue Size: ${data.queue_size}`);
    } catch (error) {
        updateScraperLog(error.message, true);
    } finally {
        toggleSpinner(false);
    }
}

async function handleProcessQueue() {
    toggleSpinner(true);
    updateScraperLog('Processing queue...');
    try {
        const response = await fetch(`${API_BASE_URL}/scrape/process`, { method: 'POST' });
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || 'Failed to process queue');
        }
        const data = await response.json();
        
        let newMarkers = [];
        if (data.inserted_properties && data.inserted_properties.length > 0) {
            data.inserted_properties.forEach(prop => {
                const marker = L.circleMarker([prop.lat, prop.long], {
                    radius: 8,
                    fillColor: '#10b981', // Emerald green
                    color: '#047857',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9,
                    className: 'live-marker'
                }).addTo(map);

                const sourceLink = prop.source_url
                    ? `<a href="${prop.source_url}" target="_blank" rel="noopener" class="scraped-link">🔗 View on Craigslist</a>`
                    : `<span class="scraped-link" style="opacity:0.5; cursor:default;">📁 Historical Dataset</span>`;

                marker.bindPopup(`
                    <h4 style="color:#10b981;">🚨 LIVE SCRAPED LISTING!</h4>
                    ${prop.source_title ? `<p style="font-size:0.8rem;color:#6B7280;margin-bottom:6px;">${prop.source_title.substring(0,60)}...</p>` : ''}
                    <p><strong>Price:</strong> ${currencyFormatter.format(prop.price)}</p>
                    <p><strong>Bedrooms:</strong> ${prop.bedrooms}</p>
                    <p><strong>Bathrooms:</strong> ${prop.bathrooms}</p>
                    <p><strong>Sqft:</strong> ${prop.sqft_living}</p>
                    <hr style="border:0; border-top:1px solid #e5e7eb; margin:8px 0;">
                    ${sourceLink}
                `);
                newMarkers.push(marker);
            });
            liveScrapedMarkerStacks.push(newMarkers);
            
            // Enforce visual limit of 1000 markers to prevent browser lag
            let totalScraped = liveScrapedMarkerStacks.reduce((sum, batch) => sum + batch.length, 0);
            while (totalScraped > 1000 && liveScrapedMarkerStacks.length > 0) {
                let oldestBatch = liveScrapedMarkerStacks.shift();
                oldestBatch.forEach(m => map.removeLayer(m));
                totalScraped -= oldestBatch.length;
            }
        }

        updateScraperLog(`${data.message}<br>Tree Size: ${data.new_tree_size}`);
    } catch (error) {
        updateScraperLog(error.message, true);
    } finally {
        toggleSpinner(false);
    }
}

async function handleUndoScrape() {
    toggleSpinner(true);
    updateScraperLog('Undoing last scrape...');
    try {
        const response = await fetch(`${API_BASE_URL}/scrape/undo`, { method: 'POST' });
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || 'Failed to undo scrape');
        }
        const data = await response.json();

        // Remove from map to match LIFO stack
        if (liveScrapedMarkerStacks.length > 0) {
            let lastBatch = liveScrapedMarkerStacks.pop();
            lastBatch.forEach(marker => map.removeLayer(marker));
        }

        updateScraperLog(`${data.message}<br>Tree Size: ${data.new_tree_size}`);
    } catch (error) {
        updateScraperLog(error.message, true);
    } finally {
        toggleSpinner(false);
    }
}

function handleClearScraped() {
    liveScrapedMarkerStacks.forEach(batch => batch.forEach(m => map.removeLayer(m)));
    liveScrapedMarkerStacks = [];
    updateScraperLog('Cleared all live scraped pins from the map.');
}

// Start app
document.addEventListener('DOMContentLoaded', init);
