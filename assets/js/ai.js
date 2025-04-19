function goToupload() {
    window.location.href = "{{ url_for('upload') }}";
  }
  
// {/* <script> */}
const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const uploadIcon = document.getElementById("upload-icon");
const fileNameDisplay = document.getElementById("file-name");
const previewContainer = document.getElementById("preview-container");
const previewImg = document.getElementById("preview-img");
const uploadText = document.getElementById("upload-text");
const submitBtn = document.getElementById("upload-btn");
const spinnerContainer = document.getElementById("spinner-container");
const statusMessage = document.getElementById("status-message");
const successMessage = document.getElementById("success-message");
const errorMessage = document.getElementById("error-message");

// Check if device is Android
const isAndroid = /Android/i.test(navigator.userAgent);

// Update text for mobile devices
if (
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
        navigator.userAgent
    )
) {
    uploadText.textContent = "Drag & Drop or Click here to Upload an Image";
}

// Handle file input change
inputFile.addEventListener("change", handleFileSelection);

// Handle file input click for Android devices
if (isAndroid) {
    dropArea.addEventListener("click", function (e) {
        // Prevent default behavior on Android devices
        if (e.target !== inputFile) {
            e.preventDefault();
            inputFile.click();
        }
    });
}

// Drag and drop functionality for desktop
if (!isAndroid) {
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ["dragenter", "dragover"].forEach((eventName) => {
        dropArea.addEventListener(
            eventName,
            () => {
                dropArea.classList.add("drag-over");
            },
            false
        );
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropArea.addEventListener(
            eventName,
            () => {
                dropArea.classList.remove("drag-over");
            },
            false
        );
    });

    dropArea.addEventListener(
        "drop",
        (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0) {
                inputFile.files = files;
                handleFileSelection();
            }
        },
        false
    );
}

// Touch events support
dropArea.addEventListener(
    "touchstart",
    (e) => {
        dropArea.classList.add("drag-over");
    },
    { passive: true }
);

dropArea.addEventListener(
    "touchend",
    () => {
        dropArea.classList.remove("drag-over");
    },
    { passive: true }
);

// File handling functions
function handleFileSelection() {
    const file = inputFile.files[0];

    if (!file) return;

    if (!file.type.match("image.*")) {
        showError("Please upload a valid image file");
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        // 5MB limit
        showError("File size too large (max 5MB)");
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        previewImg.src = e.target.result;
        dropArea.style.border = "none";
        previewContainer.style.display = "block";
        uploadIcon.style.display = "none";
        uploadText.style.display = "none";
        fileNameDisplay.textContent = file.name;
        submitBtn.style.display = "block";
        submitBtn.disabled = false;
        hideMessages();
    };
    reader.readAsDataURL(file);
}

function resetForm() {
    previewContainer.style.display = "none";
    uploadIcon.style.display = "block";
    uploadText.style.display = "block";
    fileNameDisplay.textContent = "(JPG, JPEG, PNG only)";
    submitBtn.style.display = "none";
    submitBtn.disabled = true;
    successMessage.style.display = "none";
    dropArea.style.border = "2px dashed #bbb5ff";
    inputFile.value = "";
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = "block";
    setTimeout(() => {
        errorMessage.style.display = "none";
    }, 3000);
}

function hideMessages() {
    successMessage.style.display = "none";
    errorMessage.style.display = "none";
}

// Fix for Android back button to reset the form
window.addEventListener("pageshow", function (event) {
    if (event.persisted) {
        resetForm();
    }
});

// UPLOAD FILE
function uploadFile(file) {
    // Upload the file to the backend
    let formData = new FormData();
    formData.append("file", file);
    fetch("/upload", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Add a delay of 1 second (1000 ms) before showing the alert
            setTimeout(() => {
                if (data.error) {
                    alert("Error: " + data.error);
                } else {
                    alert("Predicted Mudra: " + data.prediction);

                }
                console.log(data.prediction)
            }, 1000);
        })
        .catch(error => {
            clearInterval(interval);
            alert("Error uploading file: " + error);
        });
}

document.getElementById("upload-btn").addEventListener("click", function () {
    console.log("Upload button clicked");  // Debugging log
    // Prevent any default actions

    // Show loading state
    submitBtn.disabled = true;
    spinnerContainer.style.display = "flex";
    statusMessage.textContent = "Uploading your image...";
    hideMessages();
});

// Submit button event with additional Android handling
submitBtn.addEventListener("click", async function (e) {


    try {
        // Simulate API call (replace with actual API integration)
        await simulateApiCall();

        // Show success
        spinnerContainer.style.display = "none";
        successMessage.textContent = "Image uploaded successfully!";
        successMessage.style.display = "block";

        const file = inputFile.files[0];
        if (file) {
            console.log("File selected:", file.name);  // Debugging log
            uploadFile(file);
        } else {
            console.log("No file selected");
        }

        // Reset form after delay
        setTimeout(resetForm, 1000);
    } catch (error) {
        showError("Processing failed. Please try again.");
        submitBtn.disabled = false;
        spinnerContainer.style.display = "none";
    }
});

// Helper functions
function simulateApiCall() {
    return new Promise((resolve, reject) => {
        // Simulate network delay (1.5-3 seconds)
        const delay = 1500 + Math.random() * 1500;

        // 5% chance of failure for demonstration
        const shouldFail = Math.random() < 0.05;

        setTimeout(() => {
            if (shouldFail) {
                reject(new Error("Simulated API failure"));
            } else {
                resolve();
            }
        }, delay);
    });
}

// </script>