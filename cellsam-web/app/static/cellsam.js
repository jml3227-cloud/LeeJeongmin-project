document.addEventListener('DOMContentLoaded', function () {
  let imageUploaded = false;

  document.getElementById('fileSelectBtn').addEventListener('click', function (e) {
    e.stopPropagation();
    if (imageUploaded) {
      document.getElementById('uploadError').style.display = 'block';
      return;
    }
    document.getElementById('imageInput').click();
  });

  const uploadZone = document.getElementById('uploadZone');

  uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    this.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', function () {
    this.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    this.classList.remove('dragover');
    if (imageUploaded) {
      document.getElementById('uploadError').style.display = 'block';
      return;
    }
    const files = e.dataTransfer.files;
    const dt = new DataTransfer();
    for (let f of files) dt.items.add(f);
    document.getElementById('imageInput').files = dt.files;
    handleFiles(files);
  });

  document.getElementById('imageInput').addEventListener('change', function () {
    handleFiles(this.files);
  });

  const previewImg = document.getElementById('previewImg');
  const deleteImgBtn = document.getElementById('deleteImgBtn');

  previewImg.addEventListener('mouseenter', function () {
    this.style.outline = '2px solid rgba(0,0,0,0.6)';
    deleteImgBtn.style.display = 'block';
  });

  previewImg.addEventListener('mouseleave', function () {
    this.style.outline = 'none';
    deleteImgBtn.style.display = 'none';
  });

  deleteImgBtn.addEventListener('mouseenter', function () {
    previewImg.style.outline = '2px solid rgba(0,0,0,0.6)';
    this.style.display = 'block';
  });

  deleteImgBtn.addEventListener('mouseleave', function () {
    previewImg.style.outline = 'none';
    this.style.display = 'none';
  });

  deleteImgBtn.addEventListener('click', function () {
    previewImg.src = '';
    document.getElementById('fileName').textContent = '';
    document.getElementById('previewArea').style.display = 'none';
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('imageInput').value = '';
    imageUploaded = false;
    document.getElementById('uploadError').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('videoResultSection').style.display = 'none';
  });

  function handleFiles(files) {
    if (!files || files.length === 0) return;

    imageUploaded = true;
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;

    if (files.length === 1) {
      const reader = new FileReader();
      reader.onload = function (e) {
        const isTif = files[0].name.toLowerCase().endsWith('.tif') || files[0].name.toLowerCase().endsWith('.tiff');
        if (!isTif) {
          previewImg.src = e.target.result;
          previewImg.style.display = 'block';
        } else {
          previewImg.style.display = 'none';
        }
        document.getElementById('fileName').textContent = files[0].name;
        document.getElementById('previewArea').style.display = 'block';
      };
      reader.readAsDataURL(files[0]);
    } else {
      previewImg.style.display = 'none';
      document.getElementById('fileName').textContent = `${files.length}개 파일 선택됨`;
      document.getElementById('previewArea').style.display = 'block';
    }
  }

  document.getElementById('analyzeBtn').addEventListener('click', function () {
    const files = document.getElementById('imageInput').files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('videoResultSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    this.disabled = true;

    if (files.length === 1) {
      formData.append('image', files[0]);
      fetch('/cellsam/analyze', {
        method: 'POST',
        body: formData
      })
      .then(res => res.json())
      .then(res => {
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('originalImg').src = previewImg.src;
        document.getElementById('resultImg').src = 'data:image/png;base64,' + res.mask_image;
        document.getElementById('cellCount').textContent = res.cell_count;
        document.getElementById('avgIoU').textContent = res.avg_iou;
        document.getElementById('resultSection').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
      })
      .catch(() => {
        document.getElementById('loadingSection').style.display = 'none';
        alert('분석 중 오류가 발생했습니다.');
        document.getElementById('analyzeBtn').disabled = false;
      });
    } else {
      for (let f of files) formData.append('images', f);
      fetch('/cellsam/analyze_video', {
        method: 'POST',
        body: formData
      })
      .then(res => res.blob())
      .then(blob => {
        document.getElementById('loadingSection').style.display = 'none';
        const url = URL.createObjectURL(blob);
        document.getElementById('resultVideo').src = url;
        document.getElementById('videoResultSection').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
      })
      .catch(() => {
        document.getElementById('loadingSection').style.display = 'none';
        alert('영상 분석 중 오류가 발생했습니다.');
        document.getElementById('analyzeBtn').disabled = false;
      });
    }
  });
});