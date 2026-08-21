// Base API URL configuration (supports standalone frontend dev servers like Live Server/Vite and Flask integrated mode)
const API_BASE_URL = (window.location.protocol === 'file:' || (window.location.port !== '5000' && window.location.port !== '10000' && window.location.port !== '80' && window.location.port !== '443' && !window.location.hostname.includes('render.com') && !window.location.hostname.includes('hf.space') && !window.location.hostname.includes('trycloudflare.com'))) 
    ? (window.BOVISTA_API_URL || 'http://127.0.0.1:5000') 
    : '';

function apiUrl(endpoint) {
    if (!endpoint) return '';
    if (endpoint.startsWith('http://') || endpoint.startsWith('https://') || endpoint.startsWith('data:') || endpoint.startsWith('blob:')) return endpoint;
    if (!endpoint.startsWith('/')) endpoint = '/' + endpoint;
    return `${API_BASE_URL}${endpoint}`;
}

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const changeImageBtn = document.getElementById("change-image-btn");
    const predictBtn = document.getElementById("predict-btn");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    
    // Results & Overlay Elements
    const resultsSection = document.getElementById("results-section");
    const loadingOverlay = document.getElementById("loading-overlay");
    const sampleGrid = document.getElementById("sample-grid");
    const catalogGrid = document.getElementById("catalog-grid");
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const printDossierBtn = document.getElementById("print-dossier-btn");

    let currentImagePayload = null; // Can be a File, base64 string, or sample path

    // 1. Initialize Sample Gallery & Breed Catalog
    loadSampleImages();
    loadBreedCatalog();

    // 2. Theme Toggle
    themeToggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("theme-light");
        const isLight = document.body.classList.contains("theme-light");
        themeToggleBtn.innerHTML = isLight ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    });

    // 3. File Browse & Drag-and-Drop
    browseBtn.addEventListener("click", () => fileInput.click());
    changeImageBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (JPG, PNG, WEBP).");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            predictBtn.disabled = false;
            currentImagePayload = { type: "file", file: file, previewUrl: e.target.result };
            
            // Clear active state on sample grid
            document.querySelectorAll(".sample-item").forEach(item => item.classList.remove("active"));
        };
        reader.readAsDataURL(file);
    }

    // 4. Quick Interactive Preset Chips Handling
    const presetChips = document.querySelectorAll(".preset-chip");
    const presetSamplePaths = {
        "gir": "dataset/Cattle Breeds/Gir/Gir_1.JPG",
        "jaffarabadi": "dataset/Buffalo/Jaffarabadi/10_Most_Expensive_Buffalo_Breeds_in_the_.jpg",
        "sahiwal": "dataset/Cattle Breeds/Sahiwal/Sahiwal_1.JPG",
        "toda": "dataset/Buffalo/toda/12_Thousand_Buffalo_On_Road_Royalty_.jpg"
    };

    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const breedKey = chip.dataset.breed;
            const path = presetSamplePaths[breedKey];
            if (path) {
                const imgUrl = apiUrl(`/dataset-img/${path}`);
                imagePreview.src = imgUrl;
                dropzonePrompt.classList.add("hidden");
                previewContainer.classList.remove("hidden");
                predictBtn.disabled = false;
                currentImagePayload = {
                    type: "sample",
                    sample_path: path,
                    previewUrl: imgUrl
                };

                // Visual micro-interaction
                chip.style.transform = "scale(1.1) translateY(-3px)";
                setTimeout(() => chip.style.transform = "", 250);

                // Run automatic diagnosis for instant feedback
                runPrediction();
            }
        });
    });

    // 5. Sample Gallery Loader
    async function loadSampleImages() {
        try {
            const res = await fetch(apiUrl("/api/sample-images"));
            const data = await res.json();

            if (data.success && data.samples.length > 0) {
                sampleGrid.innerHTML = "";
                data.samples.forEach(sample => {
                    const item = document.createElement("div");
                    item.className = "sample-item";
                    const sampleImgUrl = apiUrl(sample.image_url);
                    item.innerHTML = `
                        <img src="${sampleImgUrl}" alt="${sample.breed_name}" class="sample-thumb" loading="lazy">
                        <div class="sample-info">
                            <div class="sample-title" title="${sample.breed_name}">${sample.breed_name}</div>
                            <div class="sample-tag">${sample.category}</div>
                        </div>
                    `;

                    item.addEventListener("click", () => {
                        document.querySelectorAll(".sample-item").forEach(i => i.classList.remove("active"));
                        item.classList.add("active");

                        imagePreview.src = sampleImgUrl;
                        dropzonePrompt.classList.add("hidden");
                        previewContainer.classList.remove("hidden");
                        predictBtn.disabled = false;

                        currentImagePayload = {
                            type: "sample",
                            sample_path: sample.file_path,
                            previewUrl: sampleImgUrl
                        };

                        // Automatically trigger prediction on sample selection for seamless UX
                        runPrediction();
                    });

                    sampleGrid.appendChild(item);
                });
            } else {
                sampleGrid.innerHTML = `<p style="font-size:0.8rem; color:var(--text-muted);">No samples found in dataset.</p>`;
            }
        } catch (err) {
            console.error("Error loading samples:", err);
            sampleGrid.innerHTML = `<p style="font-size:0.8rem; color:var(--accent-rose);">Failed to load samples.</p>`;
        }
    }

    // 6. Run Prediction & Diagnosis
    predictBtn.addEventListener("click", runPrediction);

    async function runPrediction() {
        if (!currentImagePayload) return;

        const scannerLaser = document.getElementById("scanner-laser");
        const scannerGrid = document.getElementById("scanner-grid");
        if (scannerLaser) scannerLaser.classList.remove("hidden");
        if (scannerGrid) scannerGrid.classList.remove("hidden");

        loadingOverlay.classList.remove("hidden");

        try {
            let response;
            if (currentImagePayload.type === "file") {
                const formData = new FormData();
                formData.append("file", currentImagePayload.file);
                response = await fetch(apiUrl("/api/predict"), {
                    method: "POST",
                    body: formData
                });
            } else if (currentImagePayload.type === "base64") {
                response = await fetch(apiUrl("/api/predict"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: currentImagePayload.data })
                });
            } else if (currentImagePayload.type === "sample") {
                response = await fetch(apiUrl("/api/predict"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_path: currentImagePayload.sample_path })
                });
            }

            const result = await response.json();
            if (result.is_blurry === true || result.error_code === "BLURRY_IMAGE" || result.error === "reupload a clear image") {
                renderPredictionResults({
                    success: false,
                    is_blurry: true,
                    is_bovine: false,
                    is_known_breed: false,
                    blur_score: result.blur_score,
                    error: "reupload a clear image",
                    alert_message: "reupload a clear image"
                });
            } else if (result.is_bovine === false || result.error === "non - bovine image detected") {
                renderPredictionResults({
                    success: false,
                    is_bovine: false,
                    is_known_breed: false,
                    error: "non - bovine image detected"
                });
            } else if (result.is_bovine === true && (result.is_known_breed === false || result.error === "the given breed does not exists in our data")) {
                renderPredictionResults({
                    success: false,
                    is_bovine: true,
                    is_known_breed: false,
                    error: "the given breed does not exists in our data"
                });
            } else if (result.success) {
                renderPredictionResults(result);
            } else {
                alert("Prediction Error: " + (result.error || "Unknown error"));
            }
        } catch (err) {
            console.error("Inference Error:", err);
            alert("Failed to connect to backend server. Please check console.");
        } finally {
            loadingOverlay.classList.add("hidden");
            if (scannerLaser) scannerLaser.classList.add("hidden");
            if (scannerGrid) scannerGrid.classList.add("hidden");
        }
    }

    // Number Counting Animation Helper
    function animateCounter(elementId, targetValue, duration = 800) {
        const el = document.getElementById(elementId);
        if (!el) return;
        const target = parseFloat(targetValue);
        if (isNaN(target)) {
            el.textContent = targetValue;
            return;
        }
        const startTime = performance.now();
        function updateCount(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            const current = (target * easeProgress).toFixed(1);
            el.textContent = current;
            if (progress < 1) {
                requestAnimationFrame(updateCount);
            } else {
                el.textContent = target.toFixed(1);
            }
        }
        requestAnimationFrame(updateCount);
    }

    // Floating Alert Notification Toast
    function showToastAlert(message, type = "warning") {
        let toast = document.getElementById("bovista-floating-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "bovista-floating-toast";
            document.body.appendChild(toast);
        }
        
        toast.className = `floating-toast ${type} active`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fa-solid ${type === 'error' ? 'fa-circle-xmark' : type === 'warning' ? 'fa-triangle-exclamation' : 'fa-circle-check'}"></i>
                <span>${message}</span>
            </div>
            <button type="button" class="toast-close-btn" onclick="this.parentElement.classList.remove('active')">&times;</button>
        `;

        setTimeout(() => {
            if (toast) toast.classList.remove("active");
        }, 4500);
    }

    let currentPredictionResult = null;
    let activeAnimalIndex = 0;
    let currentImageViewMode = "annotated"; // "annotated" or "crop"

    // 7. Render Complete Prediction & Multi-Animal Veterinary Dossier
    function renderPredictionResults(result) {
        resultsSection.classList.remove("hidden");
        currentPredictionResult = result;
        activeAnimalIndex = 0;
        currentImageViewMode = "annotated";

        const nonBovineCard = document.getElementById("non-bovine-card");
        const bovineWrapper = document.getElementById("bovine-results-wrapper");
        const multiBanner = document.getElementById("multi-animal-banner");
        const imgViewToggle = document.getElementById("img-view-toggle");
        const multiTableCard = document.getElementById("multi-animal-table-card");

        // Case 0: Blurry Image Uploaded
        if (result.is_blurry === true || result.error_code === "BLURRY_IMAGE" || result.error === "reupload a clear image") {
            if (bovineWrapper) bovineWrapper.classList.add("hidden");
            if (nonBovineCard) {
                nonBovineCard.classList.remove("hidden");
                nonBovineCard.className = "glass-card non-bovine-card warning-card-theme";

                const nonBovineImg = document.getElementById("non-bovine-img");
                if (nonBovineImg) nonBovineImg.src = imagePreview.src;

                const nonBovineTitle = document.getElementById("non-bovine-title");
                if (nonBovineTitle) nonBovineTitle.textContent = "reupload a clear image";

                const nonBovineSub = document.querySelector(".non-bovine-sub");
                if (nonBovineSub) nonBovineSub.textContent = "The uploaded bovine photo is too blurry for accurate breed classification";

                const nonBovineBadge = document.getElementById("non-bovine-category-badge");
                if (nonBovineBadge) {
                    nonBovineBadge.textContent = "BLURRY IMAGE";
                    nonBovineBadge.className = "category-badge warning-badge";
                }

                const nonBovineExp = document.getElementById("non-bovine-explanation");
                if (nonBovineExp) {
                    nonBovineExp.innerHTML = `
                        <strong>Image Clarity Alert:</strong> The uploaded image lacks sharp focus and physical details required to identify horn conformation, coat patterns, and facial structure.
                        <br><br>
                        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); padding: 0.8rem 1rem; border-radius: 8px; color: #fbbf24; font-weight: 600; font-size: 0.95rem; display: flex; align-items: center; gap: 0.6rem;">
                            <i class="fa-solid fa-triangle-exclamation" style="font-size: 1.2rem;"></i>
                            <span>Please reupload a clear image of the cattle or buffalo.</span>
                        </div>
                    `;
                }

                const btnReupload = document.getElementById("btn-reupload-bovine");
                if (btnReupload) {
                    btnReupload.innerHTML = '<i class="fa-solid fa-camera-rotate"></i> Reupload a Clear Image';
                    btnReupload.onclick = () => fileInput.click();
                }

                nonBovineCard.scrollIntoView({ behavior: "smooth" });
            }

            // Display Toast Alert
            showToastAlert("reupload a clear image", "warning");
            return;
        }

        // Case A: Non-Bovine Subject
        if (result.is_bovine === false || result.error === "non - bovine image detected") {
            if (bovineWrapper) bovineWrapper.classList.add("hidden");
            if (nonBovineCard) {
                nonBovineCard.classList.remove("hidden");
                nonBovineCard.className = "glass-card non-bovine-card error-card-theme";

                const nonBovineImg = document.getElementById("non-bovine-img");
                if (nonBovineImg) nonBovineImg.src = imagePreview.src;

                const nonBovineTitle = document.getElementById("non-bovine-title");
                if (nonBovineTitle) nonBovineTitle.textContent = "non - bovine image detected";

                const nonBovineSub = document.querySelector(".non-bovine-sub");
                if (nonBovineSub) nonBovineSub.textContent = "The uploaded image is not a cattle or buffalo";

                const nonBovineBadge = document.getElementById("non-bovine-category-badge");
                if (nonBovineBadge) {
                    nonBovineBadge.textContent = "NON-BOVINE";
                    nonBovineBadge.className = "category-badge error-badge";
                }

                const nonBovineExp = document.getElementById("non-bovine-explanation");
                if (nonBovineExp) {
                    nonBovineExp.innerHTML = `
                        <strong>non - bovine image detected:</strong> This photo is not a cattle or buffalo. Classification probabilities and veterinary analytics are not performed for non-bovine images.
                    `;
                }

                const btnReupload = document.getElementById("btn-reupload-bovine");
                if (btnReupload) {
                    btnReupload.onclick = () => fileInput.click();
                }

                nonBovineCard.scrollIntoView({ behavior: "smooth" });
            }
            return;
        }

        // Case B: Cattle / Buffalo Detected, but Breed NOT in Dataset
        if (result.is_bovine === true && (result.is_known_breed === false || result.error === "the given breed does not exists in our data")) {
            if (bovineWrapper) bovineWrapper.classList.add("hidden");
            if (nonBovineCard) {
                nonBovineCard.classList.remove("hidden");
                nonBovineCard.className = "glass-card non-bovine-card warning-card-theme";

                const nonBovineImg = document.getElementById("non-bovine-img");
                if (nonBovineImg) nonBovineImg.src = imagePreview.src;

                const nonBovineTitle = document.getElementById("non-bovine-title");
                if (nonBovineTitle) nonBovineTitle.textContent = "the given breed does not exists in our data";

                const nonBovineSub = document.querySelector(".non-bovine-sub");
                if (nonBovineSub) nonBovineSub.textContent = "Bovine subject verified, but breed is outside our 10-breed database";

                const nonBovineBadge = document.getElementById("non-bovine-category-badge");
                if (nonBovineBadge) {
                    nonBovineBadge.textContent = "UNRECOGNIZED BREED";
                    nonBovineBadge.className = "category-badge warning-badge";
                }

                const nonBovineExp = document.getElementById("non-bovine-explanation");
                if (nonBovineExp) {
                    nonBovineExp.innerHTML = `
                        <strong>the given breed does not exists in our data:</strong> The image is confirmed to be a cattle or buffalo, but its specific breed is not present in our 10-breed catalog.
                        <br><br>
                        <em>Supported Breeds:</em> Chhattisgarhi, Gir, Jaffarabadi, Jersey, Kankrej, Marathwada, Red Sindhi, Sahiwal, Surti, Toda.
                    `;
                }

                const btnReupload = document.getElementById("btn-reupload-bovine");
                if (btnReupload) {
                    btnReupload.onclick = () => fileInput.click();
                }

                nonBovineCard.scrollIntoView({ behavior: "smooth" });
            }
            return;
        }

        // Handle Confirmed Bovine Subject (Single or Multiple)
        if (nonBovineCard) nonBovineCard.classList.add("hidden");
        if (bovineWrapper) bovineWrapper.classList.remove("hidden");

        const defaultImg = result.crop_image || result.annotated_image || result.image_url || result.breed_details?.image_url || imagePreview.src;
        const instances = result.instances && result.instances.length > 0 ? result.instances : [{
            instance_id: 1,
            predicted_breed: result.predicted_breed,
            breed_details: result.breed_details,
            top_candidates: result.top_candidates,
            crop_image: defaultImg
        }];

        const isMulti = instances.length > 1;

        // Multi-Animal Banner Setup
        if (multiBanner) {
            if (isMulti) {
                multiBanner.classList.remove("hidden");
                const countSpan = document.getElementById("multi-detected-count");
                if (countSpan) countSpan.textContent = instances.length;

                const chipsRow = document.getElementById("animal-chips-row");
                if (chipsRow) {
                    chipsRow.innerHTML = "";
                    instances.forEach((inst, idx) => {
                        const chip = document.createElement("div");
                        chip.className = `animal-chip ${idx === 0 ? "active" : ""}`;
                        chip.id = `animal-chip-${idx}`;
                        const pb = inst.predicted_breed || {};
                        const bName = pb.name || inst.breed_name || `Animal #${inst.instance_id}`;
                        const confVal = pb.confidence_percent || inst.confidence_percent;

                        chip.innerHTML = `
                            <img src="${inst.crop_image || imagePreview.src}" class="chip-thumb" alt="Animal #${inst.instance_id}">
                            <div class="chip-info">
                                <span class="chip-num">Animal #${inst.instance_id}</span>
                                <span class="chip-name">${bName}</span>
                                <span class="chip-meta">${confVal ? confVal + '%' : (inst.is_bovine ? 'Bovine' : 'Other')}</span>
                            </div>
                        `;
                        chip.addEventListener("click", () => switchAnimalInstance(idx));
                        chipsRow.appendChild(chip);
                    });
                }
            } else {
                multiBanner.classList.add("hidden");
            }
        }

        // Image View Switcher Toggle
        if (imgViewToggle) {
            if (isMulti) {
                imgViewToggle.classList.remove("hidden");
            } else {
                imgViewToggle.classList.add("hidden");
            }
        }

        // Multi-Animal Comparison Table Setup
        if (multiTableCard) {
            if (isMulti) {
                multiTableCard.classList.remove("hidden");
                const tableBody = document.getElementById("comparison-table-body");
                if (tableBody) {
                    tableBody.innerHTML = "";
                    instances.forEach((inst, idx) => {
                        const tr = document.createElement("tr");
                        tr.id = `comparison-row-${idx}`;
                        if (idx === 0) tr.className = "active-row";

                        const pb = inst.predicted_breed || {};
                        const bd = inst.breed_details || {};
                        const mp = bd.milk_production || {};
                        const econ = bd.market_price || {};

                        tr.innerHTML = `
                            <td><strong>#${inst.instance_id}</strong></td>
                            <td><img src="${inst.crop_image || imagePreview.src}" class="table-img-thumb" alt="Animal #${inst.instance_id}"></td>
                            <td><strong>${pb.name || inst.breed_name || 'N/A'}</strong></td>
                            <td><span class="category-badge" style="position:static; display:inline-block;">${pb.category || 'Bovine'}</span></td>
                            <td><strong>${pb.confidence_percent ? pb.confidence_percent + '%' : 'N/A'}</strong></td>
                            <td>${mp.daily_yield_liters || 'N/A'}</td>
                            <td>${econ.currency_inr || 'N/A'}</td>
                            <td><button type="button" class="table-btn-select" onclick="window.switchAnimalInstance(${idx})"><i class="fa-solid fa-eye"></i> View Dossier</button></td>
                        `;
                        tableBody.appendChild(tr);
                    });
                }
            } else {
                multiTableCard.classList.add("hidden");
            }
        }

        // Populate initial animal instance (index 0)
        switchAnimalInstance(0);

        // Reveal Results Section with smooth scroll
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // Update Animal Image Preview (Scene vs Crop vs Catalog breed photo)
    function updateAnimalImageDisplay() {
        const resImg = document.getElementById("result-animal-img");
        if (!resImg || !currentPredictionResult) return;

        const instances = currentPredictionResult.instances && currentPredictionResult.instances.length > 0
            ? currentPredictionResult.instances
            : [{
                instance_id: 1,
                predicted_breed: currentPredictionResult.predicted_breed,
                breed_details: currentPredictionResult.breed_details,
                top_candidates: currentPredictionResult.top_candidates,
                crop_image: currentPredictionResult.crop_image || currentPredictionResult.annotated_image || currentPredictionResult.breed_details?.image_url || imagePreview.src
            }];

        const currentInst = instances[activeAnimalIndex] || instances[0];
        const breedImgUrl = currentInst.crop_image 
            || currentInst.breed_details?.image_url 
            || currentPredictionResult.annotated_image 
            || currentPredictionResult.crop_image
            || currentPredictionResult.breed_details?.image_url 
            || currentPredictionResult.image_url 
            || imagePreview.src;

        if (currentImageViewMode === "crop" && currentInst.crop_image) {
            resImg.src = currentInst.crop_image;
        } else if (currentImageViewMode === "annotated" && currentPredictionResult.annotated_image) {
            resImg.src = currentPredictionResult.annotated_image;
        } else {
            resImg.src = breedImgUrl || "/static/images/placeholder_bovine.jpg";
        }

        // Keep classifier preview dropzone in sync
        if (breedImgUrl && (!imagePreview.src || imagePreview.src.includes("blob:") || imagePreview.src.endsWith("#") || imagePreview.src.length < 5)) {
            imagePreview.src = breedImgUrl;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            predictBtn.disabled = false;
        }
    }

    // Switch active animal instance for multi-bovine scenes
    function switchAnimalInstance(index) {
        if (!currentPredictionResult) return;
        const instances = currentPredictionResult.instances && currentPredictionResult.instances.length > 0
            ? currentPredictionResult.instances
            : [{
                instance_id: 1,
                predicted_breed: currentPredictionResult.predicted_breed,
                breed_details: currentPredictionResult.breed_details,
                top_candidates: currentPredictionResult.top_candidates,
                crop_image: currentPredictionResult.crop_image || currentPredictionResult.annotated_image || currentPredictionResult.breed_details?.image_url || imagePreview.src
            }];

        if (index < 0 || index >= instances.length) return;
        activeAnimalIndex = index;

        const currentInst = instances[index];
        const breed = currentInst.predicted_breed || currentPredictionResult.predicted_breed || {
            name: currentInst.breed_name || "Bovine Subject",
            category: "Bovine",
            sub_category: "Dairy",
            origin: "N/A",
            confidence_percent: currentInst.confidence_percent || 90.0
        };
        const details = currentInst.breed_details || currentPredictionResult.breed_details || {};

        // Update active chip classes
        document.querySelectorAll(".animal-chip").forEach((c, idx) => {
            if (idx === index) c.classList.add("active");
            else c.classList.remove("active");
        });

        // Update active comparison table rows
        document.querySelectorAll("#comparison-table tbody tr").forEach((r, idx) => {
            if (idx === index) r.classList.add("active-row");
            else r.classList.remove("active-row");
        });

        // Update Animal Image Preview (Scene vs Crop)
        updateAnimalImageDisplay();

        const resImg = document.getElementById("result-animal-img");

        // Update Gallery Preview Strip if multiple images available
        const galleryStrip = document.getElementById("breed-gallery-strip");
        if (galleryStrip) {
            const galleryList = details.gallery || currentPredictionResult.gallery || [];
            if (galleryList && galleryList.length > 1) {
                galleryStrip.innerHTML = "";
                galleryStrip.classList.remove("hidden");
                galleryList.forEach((gItem, gIdx) => {
                    const thumb = document.createElement("img");
                    thumb.src = gItem.image_url;
                    thumb.className = `gallery-thumb ${gItem.image_url === (resImg ? resImg.src : '') ? "active" : ""}`;
                    thumb.alt = `${breed.name} photo ${gIdx + 1}`;
                    thumb.title = `View photo #${gIdx + 1}`;
                    thumb.addEventListener("click", (e) => {
                        e.stopPropagation();
                        if (resImg) resImg.src = gItem.image_url;
                        imagePreview.src = gItem.image_url;
                        currentImagePayload = {
                            type: "sample",
                            sample_path: gItem.file_path,
                            previewUrl: gItem.image_url
                        };
                        document.querySelectorAll(".gallery-thumb").forEach(t => t.classList.remove("active"));
                        thumb.classList.add("active");
                    });
                    galleryStrip.appendChild(thumb);
                });
            } else {
                galleryStrip.classList.add("hidden");
            }
        }

        // Update Spotlight Header
        const badgeLabel = document.getElementById("res-predicted-badge");
        if (badgeLabel) {
            badgeLabel.textContent = instances.length > 1 ? `AI PREDICTED BREED (ANIMAL #${currentInst.instance_id})` : "AI PREDICTED BREED";
        }
        document.getElementById("res-breed-name").textContent = breed.name;
        document.getElementById("res-category-badge").textContent = (breed.category || "Bovine") + " • " + (breed.sub_category || "Dairy");
        document.getElementById("res-origin-pill").innerHTML = `<i class="fa-solid fa-location-dot"></i> ${breed.origin || 'N/A'}`;
        document.getElementById("res-species-pill").innerHTML = `<i class="fa-solid fa-dna"></i> ${details.scientific_name || 'Bovine'}`;
        document.getElementById("res-subcat-pill").innerHTML = `<i class="fa-solid fa-droplet"></i> ${breed.sub_category || 'Dairy'}`;
        document.getElementById("res-breed-desc").textContent = details.description || "Comprehensive veterinary and breed classification profile.";

        // Confidence Gauge
        const confPct = typeof breed.confidence_percent === "number" ? breed.confidence_percent : 90.0;
        animateCounter("res-confidence-pct", confPct, 900);
        const gaugeBar = document.getElementById("gauge-bar");
        const totalCircumference = 264;
        const offset = totalCircumference - (totalCircumference * (confPct / 100));
        gaugeBar.style.strokeDashoffset = offset;

        // Top Candidates Probability Bars
        const candList = document.getElementById("candidates-list");
        candList.innerHTML = "";
        const cands = currentInst.top_candidates && currentInst.top_candidates.length > 0
            ? currentInst.top_candidates
            : (currentPredictionResult.top_candidates || []);

        cands.forEach(cand => {
            const candDiv = document.createElement("div");
            candDiv.className = "candidate-row";
            candDiv.innerHTML = `
                <div class="cand-meta">
                    <span>${cand.display_name}</span>
                    <span>${cand.confidence_percent}%</span>
                </div>
                <div class="cand-bar-bg">
                    <div class="cand-bar-fill" style="width: ${cand.confidence_percent}%"></div>
                </div>
            `;
            candList.appendChild(candDiv);
        });

        // Tab 1: Overview & Lifespan
        const lifespanList = document.getElementById("overview-lifespan-list");
        const ls = details.lifespan || {};
        lifespanList.innerHTML = `
            <li><span class="spec-label">Average Lifespan</span><span class="spec-val">${ls.average_lifespan_years || '15 - 20 years'}</span></li>
            <li><span class="spec-label">Productive Milking Years</span><span class="spec-val">${ls.productive_lactation_years || '10 - 14 years'}</span></li>
            <li><span class="spec-label">Age at First Calving</span><span class="spec-val">${ls.age_at_first_calving_months || '24 - 30 months'}</span></li>
            <li><span class="spec-label">Calving Interval</span><span class="spec-val">${ls.calving_interval_months || '12 - 14 months'}</span></li>
        `;

        const physicalList = document.getElementById("overview-physical-list");
        const pt = details.physical_traits || {};
        physicalList.innerHTML = `
            <li><span class="spec-label">Coat Color</span><span class="spec-val">${pt.coat_color || 'N/A'}</span></li>
            <li><span class="spec-label">Horn Conformation</span><span class="spec-val">${pt.horns || 'N/A'}</span></li>
            <li><span class="spec-label">Female Adult Weight</span><span class="spec-val">${pt.body_weight?.female_adult_kg || 'N/A'}</span></li>
            <li><span class="spec-label">Male Adult Weight</span><span class="spec-val">${pt.body_weight?.male_adult_kg || 'N/A'}</span></li>
            <li><span class="spec-label">Temperament</span><span class="spec-val">${pt.temperament || 'Docile'}</span></li>
        `;

        document.getElementById("overview-climate-text").textContent = pt.climate_resilience || "Highly adaptable to diverse regional climates.";

        // Tab 2: Milk & Quality
        const mp = details.milk_production || {};
        const mq = details.milk_quality || {};
        document.getElementById("milk-daily-yield").textContent = mp.daily_yield_liters || "18 - 25 L/day";
        document.getElementById("milk-lactation-yield").textContent = mp.lactation_yield_liters || "5,000 - 7,500 L";
        document.getElementById("milk-fat-pct").textContent = mp.fat_percentage || "4.2%";
        document.getElementById("milk-snf-val").textContent = `${mp.snf_percentage || '9.0%'} / ${mp.protein_percentage || '3.5%'}`;

        const qualityGrid = document.getElementById("milk-quality-details");
        qualityGrid.innerHTML = `
            <div class="quality-item">
                <div class="quality-title">Beta-Casein Genetics</div>
                <div class="quality-val">${mq.beta_casein_type || 'A2/A1 Blend'}</div>
            </div>
            <div class="quality-item">
                <div class="quality-title">Fat Globule Conformation</div>
                <div class="quality-val">${mq.fat_globule_size || 'Optimal Digestion Density'}</div>
            </div>
            <div class="quality-item">
                <div class="quality-title">Processing & Culinary Suitability</div>
                <div class="quality-val">${mq.suitability || 'Artisanal dairy products & drinking milk'}</div>
            </div>
            <div class="quality-item">
                <div class="quality-title">Nutritional & Bioactive Profile</div>
                <div class="quality-val">${mq.nutritional_highlights || 'Rich in essential minerals and bioactive proteins'}</div>
            </div>
        `;

        // Tab 3: Diseases, Symptoms & Cures
        const emergencyText = details.cure_and_treatment?.emergency_first_aid || "Immediately isolate animal and consult licensed veterinarian.";
        document.getElementById("health-emergency-text").textContent = emergencyText;

        const diseasesList = document.getElementById("diseases-list");
        diseasesList.innerHTML = "";
        (details.possible_diseases || []).forEach(d => {
            const dCard = document.createElement("div");
            dCard.className = "disease-card";
            dCard.innerHTML = `
                <div class="disease-header">
                    <span class="disease-name">${d.name}</span>
                    <span class="severity-tag severity-${d.severity}">${d.severity}</span>
                </div>
                <div class="disease-symptoms"><strong>Symptoms:</strong> ${d.symptoms}</div>
                <div class="disease-risk"><strong>Predisposition:</strong> ${d.risk_factors || 'General environmental factors'}</div>
            `;
            diseasesList.appendChild(dCard);
        });

        const vaxMedsList = document.getElementById("veterinary-medicines-list");
        vaxMedsList.innerHTML = "";
        (details.cure_and_treatment?.veterinary_medicines || []).forEach(med => {
            const li = document.createElement("li");
            li.innerHTML = `<span class="spec-val" style="text-align:left;">• ${med}</span>`;
            vaxMedsList.appendChild(li);
        });

        document.getElementById("ethno-remedies-text").textContent = details.cure_and_treatment?.ethnoveterinary_remedies || "Clean turmeric, aloe vera, and neem applications.";

        // Tab 4: Vaccination Schedule
        const vaxTimeline = document.getElementById("vaccination-timeline");
        vaxTimeline.innerHTML = "";
        (details.vaccination_schedule || []).forEach(vax => {
            const vDiv = document.createElement("div");
            vDiv.className = "timeline-item";
            vDiv.innerHTML = `
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <span class="vax-name">${vax.vaccine}</span>
                        <span class="vax-timing">${vax.timing}</span>
                    </div>
                    <div class="vax-route"><i class="fa-solid fa-syringe"></i> Dose & Route: ${vax.dose_route}</div>
                    <div class="vax-desc">${vax.importance}</div>
                </div>
            `;
            vaxTimeline.appendChild(vDiv);
        });

        // Tab 5: Economics & Market ROI
        const econ = details.market_price || {};
        document.getElementById("econ-market-price-inr").textContent = econ.currency_inr || "₹60,000 - ₹1,20,000";
        document.getElementById("econ-market-price-usd").textContent = `(${econ.currency_usd || '$750 - $1,500 USD'})`;
        document.getElementById("econ-roi-summary").textContent = econ.economic_roi || "High commercial profitability.";

        const priceRows = document.getElementById("econ-price-rows");
        priceRows.innerHTML = `
            <div class="price-row">
                <span class="price-type">Milking Cow / Buffalo (In Lactation)</span>
                <span class="price-amount">${econ.milking_cow_price_inr || '₹75,000 - ₹1,20,000'}</span>
            </div>
            <div class="price-row">
                <span class="price-type">Pregnant Heifer</span>
                <span class="price-amount">${econ.pregnant_heifer_price_inr || '₹50,000 - ₹80,000'}</span>
            </div>
            <div class="price-row">
                <span class="price-type">Pedigree Breeding Bull</span>
                <span class="price-amount">${econ.pedigree_bull_price_inr || '₹90,000 - ₹1,60,000'}</span>
            </div>
        `;

        // Tab 6: Feed & Maintenance
        const maint = details.maintenance_and_housing || {};
        const feed = maint.daily_feed_requirements || {};
        const feedMatrix = document.getElementById("feed-matrix-container");
        feedMatrix.innerHTML = `
            <div class="feed-cell">
                <div class="feed-cell-lbl">Green Fodder</div>
                <div class="feed-cell-val">${feed.green_fodder_kg || '25 - 35 kg/day'}</div>
            </div>
            <div class="feed-cell">
                <div class="feed-cell-lbl">Dry Fodder</div>
                <div class="feed-cell-val">${feed.dry_fodder_kg || '5 - 7 kg/day'}</div>
            </div>
            <div class="feed-cell">
                <div class="feed-cell-lbl">Concentrate Ratio</div>
                <div class="feed-cell-val">${feed.concentrate_feed_kg || '1 kg per 2.5 L yield'}</div>
            </div>
            <div class="feed-cell">
                <div class="feed-cell-lbl">Clean Drinking Water</div>
                <div class="feed-cell-val">${feed.clean_drinking_water_liters || '80 - 120 L/day'}</div>
            </div>
        `;

        document.getElementById("maint-housing-text").textContent = maint.housing_and_shed_design || "Adequate ventilation with non-slip flooring.";
        document.getElementById("maint-heat-text").textContent = maint.summer_heat_management || "Sprinklers, industrial fans, and wallowing pools.";
        document.getElementById("maint-milking-text").textContent = maint.milking_hygiene_protocol || "Strict teat dipping before and after milking.";
    }

    // Expose switchAnimalInstance globally for table buttons
    window.switchAnimalInstance = switchAnimalInstance;

    // View Toggle Buttons Event Listeners
    const btnViewAnnotated = document.getElementById("btn-view-annotated");
    const btnViewCrop = document.getElementById("btn-view-crop");

    if (btnViewAnnotated) {
        btnViewAnnotated.addEventListener("click", () => {
            currentImageViewMode = "annotated";
            btnViewAnnotated.classList.add("active");
            if (btnViewCrop) btnViewCrop.classList.remove("active");
            updateAnimalImageDisplay();
        });
    }

    if (btnViewCrop) {
        btnViewCrop.addEventListener("click", () => {
            currentImageViewMode = "crop";
            btnViewCrop.classList.add("active");
            if (btnViewAnnotated) btnViewAnnotated.classList.remove("active");
            updateAnimalImageDisplay();
        });
    }

    // 8. Tab Navigation Logic
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetTab = document.getElementById(btn.dataset.tab);
            if (targetTab) targetTab.classList.add("active");
        });
    });

    // 9. Breed Catalog Explorer
    async function loadBreedCatalog() {
        try {
            const res = await fetch(apiUrl("/api/breeds"));
            const data = await res.json();
            if (data.success) {
                renderCatalog(data.breeds);
                setupCatalogFilters(data.breeds);
            }
        } catch (err) {
            console.error("Error loading catalog:", err);
        }
    }

    function renderCatalog(breeds) {
        catalogGrid.innerHTML = "";
        breeds.forEach(breed => {
            const card = document.createElement("div");
            card.className = "catalog-card glass-card";
            
            // Clean and compact metrics
            const cleanYield = (breed.daily_yield || "N/A").split("(")[0].replace("liters/day", "L/day").trim();
            const cleanFat = (breed.fat_percentage || "N/A").split("(")[0].trim();
            const cleanLife = (breed.average_lifespan || "N/A").split("(")[0].replace("years", "yrs").trim();
            const isBuffalo = (breed.category || "").toLowerCase() === "buffalo";
            const categoryBadgeClass = isBuffalo ? "cat-badge-buffalo" : "cat-badge-cattle";
            const cardImgUrl = apiUrl(breed.image_url) || "images/Bovista.jpeg";

            card.innerHTML = `
                <div class="catalog-card-media">
                    <img src="${cardImgUrl}" alt="${breed.name}" class="catalog-card-img" loading="lazy">
                    <span class="cat-category-badge ${categoryBadgeClass}">${breed.category}</span>
                    <div class="catalog-card-overlay">
                        <span class="view-pill"><i class="fa-solid fa-wand-magic-sparkles"></i> AI Predicted Profile</span>
                    </div>
                </div>
                <div class="catalog-card-body">
                    <div class="catalog-card-header">
                        <div class="cat-breed-title">${breed.name}</div>
                    </div>
                    <div class="cat-origin-row">
                        <i class="fa-solid fa-location-dot"></i> <span>${breed.origin}</span>
                    </div>
                    <div class="cat-metrics-grid">
                        <div class="cat-metric-box">
                            <span class="metric-label"><i class="fa-solid fa-droplet"></i> Daily Yield</span>
                            <span class="metric-value">${cleanYield}</span>
                        </div>
                        <div class="cat-metric-box">
                            <span class="metric-label"><i class="fa-solid fa-percent"></i> Fat %</span>
                            <span class="metric-value">${cleanFat}</span>
                        </div>
                        <div class="cat-metric-box">
                            <span class="metric-label"><i class="fa-solid fa-hourglass-half"></i> Lifespan</span>
                            <span class="metric-value">${cleanLife}</span>
                        </div>
                    </div>
                    <div class="cat-card-action">
                        <span><i class="fa-solid fa-file-waveform"></i> View AI Breed Intelligence <i class="fa-solid fa-arrow-right"></i></span>
                    </div>
                </div>
            `;

            card.addEventListener("click", async () => {
                loadingOverlay.classList.remove("hidden");
                try {
                    const detailRes = await fetch(apiUrl(`/api/breed/${breed.id}`));
                    const detailData = await detailRes.json();
                    if (detailData.success) {
                        const breedDetail = detailData.breed;
                        const imgUrl = apiUrl(breedDetail.image_url || breed.image_url);
                        const filePath = breedDetail.file_path || breed.file_path;

                        // Synchronize input dropzone preview
                        if (imgUrl) {
                            imagePreview.src = imgUrl;
                            dropzonePrompt.classList.add("hidden");
                            previewContainer.classList.remove("hidden");
                            predictBtn.disabled = false;
                            currentImagePayload = {
                                type: "sample",
                                sample_path: filePath,
                                previewUrl: imgUrl
                            };
                        }

                        renderPredictionResults({
                            success: true,
                            is_bovine: true,
                            is_known_breed: true,
                            total_detected: 1,
                            crop_image: imgUrl,
                            annotated_image: imgUrl,
                            image_url: imgUrl,
                            gallery: breedDetail.gallery || breed.gallery || [],
                            predicted_breed: {
                                id: breed.id,
                                name: breed.name,
                                category: breed.category,
                                sub_category: breed.sub_category,
                                confidence_percent: 100.0,
                                origin: breed.origin
                            },
                            top_candidates: [
                                { display_name: breed.name, confidence_percent: 100.0 }
                            ],
                            breed_details: breedDetail,
                            instances: [{
                                instance_id: 1,
                                crop_image: imgUrl,
                                predicted_breed: {
                                    id: breed.id,
                                    name: breed.name,
                                    category: breed.category,
                                    sub_category: breed.sub_category,
                                    confidence_percent: 100.0,
                                    origin: breed.origin
                                },
                                top_candidates: [
                                    { display_name: breed.name, confidence_percent: 100.0 }
                                ],
                                breed_details: breedDetail
                            }]
                        });
                    }
                } catch (e) {
                    console.error("Error loading breed details:", e);
                } finally {
                    loadingOverlay.classList.add("hidden");
                }
            });

            catalogGrid.appendChild(card);
        });
    }

    function setupCatalogFilters(allBreeds) {
        const cattleCount = allBreeds.filter(b => b.category.toLowerCase() === "cattle").length;
        const buffaloCount = allBreeds.filter(b => b.category.toLowerCase() === "buffalo").length;

        const filterBtns = document.querySelectorAll(".filter-btn");
        filterBtns.forEach(btn => {
            const filter = btn.dataset.filter;
            if (filter === "all") btn.textContent = `All (${allBreeds.length})`;
            else if (filter.toLowerCase() === "cattle") btn.textContent = `Cattle Breeds (${cattleCount})`;
            else if (filter.toLowerCase() === "buffalo") btn.textContent = `Buffalo Breeds (${buffaloCount})`;

            btn.addEventListener("click", () => {
                filterBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                if (filter === "all") {
                    renderCatalog(allBreeds);
                } else {
                    const filtered = allBreeds.filter(b => b.category.toLowerCase() === filter.toLowerCase());
                    renderCatalog(filtered);
                }
            });
        });
    }

    // 10. Print / Save Dossier Action
    printDossierBtn.addEventListener("click", () => {
        window.print();
    });

    // 11. Mobile QR Code Browser Connector
    const downloadMobileBtn = document.getElementById("download-mobile-btn");
    const mobileDownloadModal = document.getElementById("mobile-download-modal");
    const closeDownloadModalBtn = document.getElementById("close-download-modal-btn");
    const closeModalOverlay = document.getElementById("close-modal-overlay");
    const qrCodeContainer = document.getElementById("qr-code-container");
    const mobileNetworkUrl = document.getElementById("mobile-network-url");
    const copyUrlBtn = document.getElementById("copy-url-btn");
    const openMobileLinkBtn = document.getElementById("open-mobile-link-btn");

    let qrCodeInstance = null;

    // Fetch and initialize Mobile Connection Details & Render QR Code
    async function initMobileDownloadInfo() {
        try {
            let mobileUrl = window.location.origin;
            if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
                const res = await fetch(apiUrl("/api/network-info"));
                const data = await res.json();
                if (data.mobile_url) {
                    mobileUrl = data.mobile_url;
                }
            }
            
            if (mobileNetworkUrl) {
                mobileNetworkUrl.value = mobileUrl;
            }
            if (openMobileLinkBtn) {
                openMobileLinkBtn.href = mobileUrl;
            }

            // Render crisp, high-resolution QR code directly in client
            if (qrCodeContainer) {
                qrCodeContainer.innerHTML = "";
                if (typeof QRCode !== "undefined") {
                    qrCodeInstance = new QRCode(qrCodeContainer, {
                        text: mobileUrl,
                        width: 170,
                        height: 170,
                        colorDark: "#0a0f18",
                        colorLight: "#ffffff",
                        correctLevel: QRCode.CorrectLevel.M
                    });
                } else {
                    // Fallback if library loading
                    const qrImg = document.createElement("img");
                    qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=170x170&data=${encodeURIComponent(mobileUrl)}&bgcolor=ffffff&color=0a0f18&margin=4`;
                    qrImg.alt = "Bovista Mobile QR Code";
                    qrImg.style.width = "170px";
                    qrImg.style.height = "170px";
                    qrCodeContainer.appendChild(qrImg);
                }
            }
        } catch (err) {
            console.error("Error initializing mobile QR info:", err);
            const fallbackUrl = window.location.origin;
            if (mobileNetworkUrl) mobileNetworkUrl.value = fallbackUrl;
            if (openMobileLinkBtn) openMobileLinkBtn.href = fallbackUrl;
            if (qrCodeContainer && typeof QRCode !== "undefined") {
                qrCodeContainer.innerHTML = "";
                new QRCode(qrCodeContainer, {
                    text: fallbackUrl,
                    width: 170,
                    height: 170,
                    colorDark: "#0a0f18",
                    colorLight: "#ffffff"
                });
            }
        }
    }

    initMobileDownloadInfo();

    // Open/Close Mobile Modal
    if (downloadMobileBtn) {
        downloadMobileBtn.addEventListener("click", () => {
            initMobileDownloadInfo();
            mobileDownloadModal.classList.remove("hidden");
        });
    }

    function closeMobileModal() {
        if (mobileDownloadModal) {
            mobileDownloadModal.classList.add("hidden");
        }
    }

    if (closeDownloadModalBtn) closeDownloadModalBtn.addEventListener("click", closeMobileModal);
    if (closeModalOverlay) closeModalOverlay.addEventListener("click", closeMobileModal);

    // Copy Mobile URL
    if (copyUrlBtn) {
        copyUrlBtn.addEventListener("click", () => {
            if (!mobileNetworkUrl) return;
            mobileNetworkUrl.select();
            mobileNetworkUrl.setSelectionRange(0, 99999);
            navigator.clipboard.writeText(mobileNetworkUrl.value).then(() => {
                const orig = copyUrlBtn.innerHTML;
                copyUrlBtn.innerHTML = '<i class="fa-solid fa-check" style="color:var(--primary);"></i>';
                setTimeout(() => { copyUrlBtn.innerHTML = orig; }, 2000);
            }).catch(() => {
                alert("URL copied: " + mobileNetworkUrl.value);
            });
        });
    }

    // 12. Android Bottom Navigation Interactions
    const bnavItems = document.querySelectorAll(".android-nav-item");
    const bnavAppBtn = document.getElementById("bnav-app-btn");

    bnavItems.forEach(item => {
        item.addEventListener("click", (e) => {
            if (item.id === "bnav-app-btn") {
                e.preventDefault();
                initMobileDownloadInfo();
                mobileDownloadModal.classList.remove("hidden");
                return;
            }

            bnavItems.forEach(b => b.classList.remove("active"));
            item.classList.add("active");
        });
    });

    if (bnavAppBtn) {
        bnavAppBtn.addEventListener("click", () => {
            initMobileDownloadInfo();
            mobileDownloadModal.classList.remove("hidden");
        });
    }

    // Update bottom nav active state based on scroll position
    const sectionIds = ["classifier-section", "sample-gallery-section", "catalog-section"];
    window.addEventListener("scroll", () => {
        let currentSection = "";
        const scrollPos = window.scrollY + 200;

        sectionIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.offsetTop <= scrollPos) {
                currentSection = id;
            }
        });

        if (currentSection) {
            bnavItems.forEach(item => {
                if (item.getAttribute("href") === `#${currentSection}`) {
                    bnavItems.forEach(b => b.classList.remove("active"));
                    item.classList.add("active");
                }
            });
        }
    }, { passive: true });

});

