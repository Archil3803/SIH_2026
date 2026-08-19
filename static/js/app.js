document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const cameraBtn = document.getElementById("camera-btn");
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
    
    // Camera Modal Elements
    const cameraModal = document.getElementById("camera-modal");
    const closeCameraBtn = document.getElementById("close-camera-btn");
    const cameraVideo = document.getElementById("camera-video");
    const capturePhotoBtn = document.getElementById("capture-photo-btn");
    const cameraCanvas = document.getElementById("camera-canvas");

    let currentImagePayload = null; // Can be a File, base64 string, or sample path
    let cameraStream = null;

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

    // 4. Live Camera Handling
    cameraBtn.addEventListener("click", async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            cameraVideo.srcObject = cameraStream;
            cameraModal.classList.remove("hidden");
        } catch (err) {
            console.error("Camera access error:", err);
            alert("Unable to access camera. Please check camera permissions or browse a photo.");
        }
    });

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraModal.classList.add("hidden");
    }

    closeCameraBtn.addEventListener("click", stopCamera);

    capturePhotoBtn.addEventListener("click", () => {
        if (!cameraVideo.videoWidth) return;

        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        const ctx = cameraCanvas.getContext("2d");
        ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        const photoBase64 = cameraCanvas.toDataURL("image/jpeg", 0.95);
        imagePreview.src = photoBase64;
        dropzonePrompt.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        predictBtn.disabled = false;
        currentImagePayload = { type: "base64", data: photoBase64 };

        stopCamera();
    });

    // 5. Sample Gallery Loader
    async function loadSampleImages() {
        try {
            const res = await fetch("/api/sample-images");
            const data = await res.json();

            if (data.success && data.samples.length > 0) {
                sampleGrid.innerHTML = "";
                data.samples.forEach(sample => {
                    const item = document.createElement("div");
                    item.className = "sample-item";
                    item.innerHTML = `
                        <img src="${sample.image_url}" alt="${sample.breed_name}" class="sample-thumb" loading="lazy">
                        <div class="sample-info">
                            <div class="sample-title" title="${sample.breed_name}">${sample.breed_name}</div>
                            <div class="sample-tag">${sample.category}</div>
                        </div>
                    `;

                    item.addEventListener("click", () => {
                        document.querySelectorAll(".sample-item").forEach(i => i.classList.remove("active"));
                        item.classList.add("active");

                        imagePreview.src = sample.image_url;
                        dropzonePrompt.classList.add("hidden");
                        previewContainer.classList.remove("hidden");
                        predictBtn.disabled = false;

                        currentImagePayload = {
                            type: "sample",
                            sample_path: sample.file_path,
                            previewUrl: sample.image_url
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

        loadingOverlay.classList.remove("hidden");

        try {
            let response;
            if (currentImagePayload.type === "file") {
                const formData = new FormData();
                formData.append("file", currentImagePayload.file);
                response = await fetch("/api/predict", {
                    method: "POST",
                    body: formData
                });
            } else if (currentImagePayload.type === "base64") {
                response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: currentImagePayload.data })
                });
            } else if (currentImagePayload.type === "sample") {
                response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_path: currentImagePayload.sample_path })
                });
            }

            const result = await response.json();
            if (result.is_bovine === false || result.error === "non - bovine image detected") {
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
        }
    }

    // 7. Render Complete Prediction & Veterinary Dossier
    function renderPredictionResults(result) {
        resultsSection.classList.remove("hidden");

        const nonBovineCard = document.getElementById("non-bovine-card");
        const bovineWrapper = document.getElementById("bovine-results-wrapper");

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
                        <em>Supported Breeds:</em> Ayrshire, Brown Swiss, Chhattisgarhi, Holstein Friesian, Jaffarabadi, Jersey, Marathwada, Red Dane, Surti, Toda.
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

        // Handle Confirmed Bovine Subject
        if (nonBovineCard) nonBovineCard.classList.add("hidden");
        if (bovineWrapper) bovineWrapper.classList.remove("hidden");

        const breed = result.predicted_breed;
        const details = result.breed_details;

        // Populate Spotlight Hero
        document.getElementById("result-animal-img").src = imagePreview.src;
        document.getElementById("res-breed-name").textContent = breed.name;
        document.getElementById("res-category-badge").textContent = breed.category + " • " + breed.sub_category;
        document.getElementById("res-origin-pill").innerHTML = `<i class="fa-solid fa-location-dot"></i> ${breed.origin}`;
        document.getElementById("res-species-pill").innerHTML = `<i class="fa-solid fa-dna"></i> ${details.scientific_name || 'Bovine'}`;
        document.getElementById("res-subcat-pill").innerHTML = `<i class="fa-solid fa-droplet"></i> ${breed.sub_category}`;
        document.getElementById("res-breed-desc").textContent = details.description || "Comprehensive breed classification profile.";

        // Confidence Gauge
        const confPct = breed.confidence_percent;
        document.getElementById("res-confidence-pct").textContent = confPct.toFixed(1);
        const gaugeBar = document.getElementById("gauge-bar");
        const totalCircumference = 264;
        const offset = totalCircumference - (totalCircumference * (confPct / 100));
        gaugeBar.style.strokeDashoffset = offset;

        // Top Candidates Probability Bars
        const candList = document.getElementById("candidates-list");
        candList.innerHTML = "";
        (result.top_candidates || []).forEach(cand => {
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

        // Reveal Results Section with smooth scroll
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });
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
            const res = await fetch("/api/breeds");
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
            card.className = "catalog-card";
            card.innerHTML = `
                <div class="catalog-top">
                    <div class="cat-breed-name">${breed.name}</div>
                    <span class="cat-category-tag">${breed.category}</span>
                </div>
                <div class="cat-origin"><i class="fa-solid fa-location-dot"></i> ${breed.origin}</div>
                <div class="cat-stats-row">
                    <div><strong>Yield:</strong> ${breed.daily_yield}</div>
                    <div><strong>Fat %:</strong> ${breed.fat_percentage}</div>
                    <div><strong>Life:</strong> ${breed.average_lifespan}</div>
                </div>
            `;

            card.addEventListener("click", async () => {
                loadingOverlay.classList.remove("hidden");
                try {
                    const detailRes = await fetch(`/api/breed/${breed.id}`);
                    const detailData = await detailRes.json();
                    if (detailData.success) {
                        renderPredictionResults({
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
                            breed_details: detailData.breed
                        });
                    }
                } catch (e) {
                    console.error(e);
                } finally {
                    loadingOverlay.classList.add("hidden");
                }
            });

            catalogGrid.appendChild(card);
        });
    }

    function setupCatalogFilters(allBreeds) {
        const filterBtns = document.querySelectorAll(".filter-btn");
        filterBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                filterBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                const filter = btn.dataset.filter;
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

    // 11. Mobile App Download & PWA System
    const downloadMobileBtn = document.getElementById("download-mobile-btn");
    const mobileDownloadModal = document.getElementById("mobile-download-modal");
    const closeDownloadModalBtn = document.getElementById("close-download-modal-btn");
    const closeModalOverlay = document.getElementById("close-modal-overlay");
    const mobileQrImg = document.getElementById("mobile-qr-img");
    const mobileNetworkUrl = document.getElementById("mobile-network-url");
    const copyUrlBtn = document.getElementById("copy-url-btn");
    const pwaInstallBtn = document.getElementById("pwa-install-btn");
    const platTabBtns = document.querySelectorAll(".plat-tab-btn");
    const androidInstallBanner = document.getElementById("android-install-banner");
    const bannerInstallBtn = document.getElementById("banner-install-btn");
    const bannerCloseBtn = document.getElementById("banner-close-btn");

    let deferredPrompt = null;

    // Register Service Worker
    if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => {
            navigator.serviceWorker.register("/sw.js")
                .then(reg => console.log("Bovista ServiceWorker registered successfully.", reg.scope))
                .catch(err => console.log("ServiceWorker registration failed:", err));
        });
    }

    // Capture beforeinstallprompt for native-like 1-click install on Android Chrome
    window.addEventListener("beforeinstallprompt", (e) => {
        e.preventDefault();
        deferredPrompt = e;
        if (pwaInstallBtn) {
            pwaInstallBtn.innerHTML = '<i class="fa-brands fa-android"></i> <span>Install Bovista App on Android</span>';
        }
        if (androidInstallBanner && !sessionStorage.getItem("bovista_banner_dismissed")) {
            androidInstallBanner.classList.remove("hidden");
        }
    });

    window.addEventListener("appinstalled", () => {
        console.log("Bovista installed successfully as native Android WebAPK.");
        if (androidInstallBanner) androidInstallBanner.classList.add("hidden");
        deferredPrompt = null;
    });

    // Handle floating install banner
    if (bannerCloseBtn) {
        bannerCloseBtn.addEventListener("click", () => {
            if (androidInstallBanner) androidInstallBanner.classList.add("hidden");
            sessionStorage.setItem("bovista_banner_dismissed", "true");
        });
    }

    async function triggerPwaInstall() {
        if (deferredPrompt) {
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            console.log(`User install outcome: ${outcome}`);
            deferredPrompt = null;
            if (androidInstallBanner) androidInstallBanner.classList.add("hidden");
        } else {
            // Open instructions modal with Android tab active
            initMobileDownloadInfo();
            mobileDownloadModal.classList.remove("hidden");
            const androidTab = document.querySelector('.plat-tab-btn[data-platform="android"]');
            if (androidTab) androidTab.click();
        }
    }

    if (pwaInstallBtn) pwaInstallBtn.addEventListener("click", triggerPwaInstall);
    if (bannerInstallBtn) bannerInstallBtn.addEventListener("click", triggerPwaInstall);

    // Fetch and initialize Mobile Connection Details
    async function initMobileDownloadInfo() {
        try {
            let mobileUrl = window.location.origin;
            if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
                const res = await fetch("/api/network-info");
                const data = await res.json();
                if (data.mobile_url) {
                    mobileUrl = data.mobile_url;
                }
            }
            
            if (mobileNetworkUrl) {
                mobileNetworkUrl.value = mobileUrl;
            }

            // Generate high-resolution QR code
            if (mobileQrImg) {
                mobileQrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(mobileUrl)}&bgcolor=ffffff&color=0a0f18&margin=4`;
            }
        } catch (err) {
            const fallbackUrl = window.location.origin;
            if (mobileNetworkUrl) mobileNetworkUrl.value = fallbackUrl;
            if (mobileQrImg) {
                mobileQrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(fallbackUrl)}`;
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

    // Platform Tab Switching (Android vs iOS)
    platTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            platTabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const platform = btn.dataset.platform;
            document.querySelectorAll(".plat-guide-content").forEach(c => c.classList.remove("active"));
            const targetGuide = document.getElementById(`guide-${platform}`);
            if (targetGuide) targetGuide.classList.add("active");
        });
    });

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

