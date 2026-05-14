$(document).ready(function () {
  let imageUploaded = false;

  // 업로드 존 클릭 → 파일 선택
  $('#fileSelectBtn').on('click', function (e) {
    e.stopPropagation();
    if (imageUploaded) {
      $('#uploadError').show();
      return;
    }
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
    if (imageUploaded) {
      $('#uploadError').show();
      return;
    }
    const file = e.originalEvent.dataTransfer.files[0];
    const dt = new DataTransfer();
    dt.items.add(file);
    $('#imageInput')[0].files = dt.files;

    handleFile(file);
  });

  // 파일 선택
  $('#imageInput').on('change', function () {
    const file = this.files[0];
    handleFile(file);
  });

  // 미리보기 hover -> x 버튼
  $('#previewImg').on('mouseenter', function() {
    $(this).css('outline', '2px solid rgba(0,0,0,0.6)');
    $('#deleteImgBtn').show();
  }).on('mouseleave', function() {
    $(this).css('outline', 'none');
    $('#deleteImgBtn').hide();
  });

  $('#deleteImgBtn').on('mouseenter', function() {
    $('#previewImg').css('outline', '2px solid rgba(0,0,0,0.6)');
    $(this).show();
  }).on('mouseleave', function() {
    $('#previewImg').css('outline', 'none');
    $(this).hide();
  });

  // x 버튼 클릭 -> 초기화
  $('#deleteImgBtn').on('click', function() {
    $('#previewImg').attr('src', '');
    $('#fileName').text('');
    $('#previewArea').hide();
    $('#uploadZone').show();
    $('#analyzeBtn').prop('disabled', true);
    $('#imageInput').val('');
    imageUploaded = false;

  });

  // 파일 처리 (미리보기)
  function handleFile(file) {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      $('#previewImg').attr('src', e.target.result);
      $('#fileName').text(file.name);
      $('#uploadZone').hide();
      $('#previewArea').show();
      $('#analyzeBtn').prop('disabled', false);
    };
    imageUploaded = true;

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
        $('#avgConfidence').text(res.avg_confidence);
      },
      error: function () {
        $('#loadingSection').hide();
        alert('분석 중 오류가 발생했습니다.');
        $('#analyzeBtn').prop('disabled', false);
      }
    });
  });

});