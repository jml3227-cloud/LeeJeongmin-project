$(document).ready(function () {

  // 업로드 존 클릭 → 파일 선택
  $('#fileSelectBtn').on('click', function (e) {
    e.stopPropagation();
    $('#imageInput').click();
  });

  // 드래그앤드롭
  $('#uploadZone').on('dragover', function (e) {
    e.preventDefault();
    $(this).addClass('dragover');
  });

  $('#uploadZone').on('dragleave', function () {
    $(this).removeClass('dragover');
  });

  $('#uploadZone').on('drop', function (e) {
    e.preventDefault();
    $(this).removeClass('dragover');
    const file = e.originalEvent.dataTransfer.files[0];
    handleFile(file);
  });

  // 파일 선택
  $('#imageInput').on('change', function () {
    const file = this.files[0];
    handleFile(file);
  });

  // 파일 처리 (미리보기)
  function handleFile(file) {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      $('#previewImg').attr('src', e.target.result);
      $('#fileName').text(file.name);
      $('#previewArea').show();
      $('#analyzeBtn').prop('disabled', false);
    };
    reader.readAsDataURL(file);
  }

  // 분석 버튼
  $('#analyzeBtn').on('click', function () {
    const file = $('#imageInput')[0].files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    $('#resultSection').hide();
    $('#loadingSection').show();
    $('#analyzeBtn').prop('disabled', true);

    $.ajax({
      url: '/cellsam/analyze',
      method: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      success: function (res) {
        $('#loadingSection').hide();
        $('#originalImg').attr('src', $('#previewImg').attr('src'));
        $('#resultImg').attr('src', 'data:image/png;base64,' + res.mask_image);
        $('#cellCount').text(res.cell_count);
        $('#resultSection').show();
        $('#analyzeBtn').prop('disabled', false);
      },
      error: function () {
        $('#loadingSection').hide();
        alert('분석 중 오류가 발생했습니다.');
        $('#analyzeBtn').prop('disabled', false);
      }
    });
  });

});