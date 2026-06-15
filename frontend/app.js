document.addEventListener("DOMContentLoaded", () => {
  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("fileInput");
  const uploadPrompt = document.getElementById("uploadPrompt");
  const imagePreview = document.getElementById("imagePreview");
  const previewImg = document.getElementById("previewImg");
  const removeBtn = document.getElementById("removeBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const loader = document.getElementById("loader");
  const resultContainer = document.getElementById("resultContainer");
  const diseaseClass = document.getElementById("diseaseClass");
  const confidenceScore = document.getElementById("confidenceScore");

  let selectedFile = null;

  // --- Drag and Drop Listeners ---
  uploadArea.addEventListener("click", () => fileInput.click());

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // --- Handle File and UI Updates ---
  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }

    selectedFile = file;

    // Read file for preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      uploadPrompt.style.display = "none";
      imagePreview.style.display = "flex";
      analyzeBtn.disabled = false;
      resultContainer.style.display = "none"; // Hide old results
    };
    reader.readAsDataURL(file);
  }

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation(); // Prevent triggering the upload area click
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    uploadPrompt.style.display = "block";
    imagePreview.style.display = "none";
    analyzeBtn.disabled = true;
    resultContainer.style.display = "none";
  });

  // --- API Request to Node.js Gateway ---
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // UI updates for loading state
    analyzeBtn.disabled = true;
    loader.style.display = "block";
    resultContainer.style.display = "none";

    // Prepare form data
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      // Send to our Node.js Server running on port 3000
      const response = await fetch("http://localhost:3000/api/analyze", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        // Format the string: 'Tomato___Early_blight' -> 'Tomato: Early blight'
        const formattedClass = result.data.class
          .replace("___", ": ")
          .replace(/_/g, " ");

        diseaseClass.textContent = formattedClass;
        confidenceScore.textContent = `${result.data.confidence}%`;
        resultContainer.style.display = "block";
      } else {
        alert("Analysis failed: " + (result.error || "Unknown error"));
      }
    } catch (error) {
      console.error("Error:", error);
      alert(
        "Failed to connect to the server. Make sure your Node backend is running!",
      );
    } finally {
      analyzeBtn.disabled = false;
      loader.style.display = "none";
    }
  });
});
