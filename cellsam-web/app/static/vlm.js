document.addEventListener('DOMContentLoaded', function () {
  const uploadZone = document.getElementById('uploadZone');
  const imageInput = document.getElementById('imageInput');
  const fileSelectBtn = document.getElementById('fileSelectBtn');
  const previewArea = document.getElementById('previewArea');
  const previewImg = document.getElementById('previewImg');
  const deleteImgBtn = document.getElementById('deleteImgBtn');
  const fileName = document.getElementById('fileName');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadError = document.getElementById('uploadError');
  const uploadSection = document.getElementById('uploadSection');
  const loadingSection = document.getElementById('loadingSection');
  const resultSection = document.getElementById('resultSection');
  const chatHistory = document.getElementById('chatHistory');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const resetBtn = document.getElementById('resetBtn');
  const chatLoadingSection = document.getElementById('chatLoadingSection');

  let selectedFile = null;

  // ── 업로드 ──────────────────────────────────────────────

  uploadZone.addEventListener('click', function () {
    if (selectedFile) { uploadError.style.display = 'block'; return; }
    imageInput.click();
  });

  fileSelectBtn.addEventListener('click', function () {
    if (selectedFile) { uploadError.style.display = 'block'; return; }
    imageInput.click();
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

  imageInput.addEventListener('change', function () {
    if (imageInput.files.length > 0) handleFile(imageInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    uploadError.style.display = 'none';

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      document.getElementById('originalImg').src = e.target.result;
    };
    reader.readAsDataURL(file);

    fileName.textContent = file.name;
    previewArea.style.display = 'block';
    deleteImgBtn.style.display = 'block';
    analyzeBtn.disabled = false;
    resultSection.style.display = 'none';
  }

  deleteImgBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    previewArea.style.display = 'none';
    previewImg.src = '';
    fileName.textContent = '';
    analyzeBtn.disabled = true;
    uploadError.style.display = 'none';
  }

  // ── 첫 번째 분석 ────────────────────────────────────────

  analyzeBtn.addEventListener('click', function () {
    if (!selectedFile) return;

    const question = document.getElementById('firstQuestion').value.trim();
    const formData = new FormData();
    formData.append('image', selectedFile);
    if (question) formData.append('question', question);

    loadingSection.style.display = 'block';
    uploadSection.style.display = 'none';
    analyzeBtn.disabled = true;

    fetch('/vlm/analyze', {
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

        // CellSAM 결과 표시
        document.getElementById('maskImg').src = 'data:image/png;base64,' + data.mask_image;

        // 첫 턴 채팅 표시
        chatHistory.innerHTML = '';
        const displayQuestion = question || '이 조직 슬라이드 소견을 말해주세요.';
        appendMessage('user', displayQuestion);
        appendMessage('assistant', data.answer);

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

  // ── 추가 질문 (2턴 이후) ────────────────────────────────

  sendBtn.addEventListener('click', sendChat);

  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  function sendChat() {
    const question = chatInput.value.trim();
    if (!question) return;

    appendMessage('user', question);
    chatInput.value = '';
    sendBtn.disabled = true;
    chatLoadingSection.style.display = 'block';

    fetch('/vlm/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question })
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        chatLoadingSection.style.display = 'none';
        sendBtn.disabled = false;

        if (data.error) {
          alert('오류: ' + data.error);
          return;
        }
        appendMessage('assistant', data.answer);
      })
      .catch(function (err) {
        chatLoadingSection.style.display = 'none';
        sendBtn.disabled = false;
        alert('서버 연결 오류: ' + err.message);
      });
  }

  function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = role === 'user' ? 'text-right mb-3' : 'text-left mb-3';

    const bubble = document.createElement('div');
    bubble.style.cssText = role === 'user'
      ? 'display:inline-block; background:#007bff; color:white; padding:10px 14px; border-radius:12px 12px 2px 12px; max-width:80%; text-align:left; white-space:pre-wrap;'
      : 'display:inline-block; background:#f1f3f5; color:#333; padding:10px 14px; border-radius:12px 12px 12px 2px; max-width:80%; text-align:left; white-space:pre-wrap;';
    bubble.textContent = text;

    div.appendChild(bubble);
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  // ── 리셋 ────────────────────────────────────────────────

  resetBtn.addEventListener('click', function () {
    resultSection.style.display = 'none';
    uploadSection.style.display = 'block';
    chatHistory.innerHTML = '';
    document.getElementById('firstQuestion').value = '';
    resetUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});