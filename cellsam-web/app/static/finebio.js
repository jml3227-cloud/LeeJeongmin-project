document.addEventListener('DOMContentLoaded', function () {
  const uploadZone = document.getElementById('uploadZone');
  const videoInput = document.getElementById('videoInput');
  const fileSelectBtn = document.getElementById('fileSelectBtn');
  const previewArea = document.getElementById('previewArea');
  const previewVideo = document.getElementById('previewVideo');
  const deleteVideoBtn = document.getElementById('deleteVideoBtn');
  const fileName = document.getElementById('fileName');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadError = document.getElementById('uploadError');
  const uploadSection = document.getElementById('uploadSection');
  const loadingSection = document.getElementById('loadingSection');
  const resultSection = document.getElementById('resultSection');
  const resetBtn = document.getElementById('resetBtn');

  let selectedFile = null;

  // ── 업로드 ──────────────────────────────────────────────

  uploadZone.addEventListener('click', function () {
    if (selectedFile) { uploadError.style.display = 'block'; return; }
    videoInput.click();
  });

  fileSelectBtn.addEventListener('click', function () {
    if (selectedFile) { uploadError.style.display = 'block'; return; }
    videoInput.click();
  });

  uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    uploadZone.style.borderColor = '#007bff';
  });

  uploadZone.addEventListener('dragleave', function () {
    uploadZone.style.borderColor = '';
  });

  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadZone.style.borderColor = '';
    if (selectedFile) { uploadError.style.display = 'block'; return; }
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  videoInput.addEventListener('change', function () {
    if (videoInput.files.length > 0) handleFile(videoInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    uploadError.style.display = 'none';

    const url = URL.createObjectURL(file);
    previewVideo.src = url;

    fileName.textContent = file.name;
    previewArea.style.display = 'block';
    analyzeBtn.disabled = false;
    resultSection.style.display = 'none';
  }

  deleteVideoBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    videoInput.value = '';
    previewArea.style.display = 'none';
    previewVideo.src = '';
    fileName.textContent = '';
    analyzeBtn.disabled = true;
    uploadError.style.display = 'none';
  }

  // ── 분석 ────────────────────────────────────────────────

  analyzeBtn.addEventListener('click', function () {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    loadingSection.style.display = 'block';
    uploadSection.style.display = 'none';
    analyzeBtn.disabled = true;

    fetch('/finebio/analyze', {
      method: 'POST',
      body: formData
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        loadingSection.style.display = 'none';

        if (data.error) {
          alert('오류: ' + data.error);
          uploadSection.style.display = 'block';
          analyzeBtn.disabled = false;
          return;
        }
        
        document.getElementById('resultVideo').src = URL.createObjectURL(selectedFile);
        
        document.getElementById('taskName').textContent = data.task_name;
        document.getElementById('completionRate').textContent = data.completion_rate;

        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(function (err) {
        loadingSection.style.display = 'none';
        uploadSection.style.display = 'block';
        analyzeBtn.disabled = false;
        alert('서버 연결 오류: ' + err.message);
      });
  });

  // ── 리셋 ────────────────────────────────────────────────

  resetBtn.addEventListener('click', function () {
    resultSection.style.display = 'none';
    uploadSection.style.display = 'block';
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});