$(document).ready(function () {
  let imageUploaded = false;

  $('#fileSelectBtn').on('click', function (e) {
    e.stopPropagation();
    if (imageUploaded) {
      $('#uploadError').show();
      return;
    }
    $('#imageInput').click();
  });

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
    const files = e.originalEvent.dataTransfer.files;
    const dt = new DataTransfer();
    for (let f of files) dt.items.add(f);
    $('#imageInput')[0].files = dt.files;
    handleFiles(files);
  });

  $('#imageInput').on('change', function () {
    handleFiles(this.files);
  });

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

  $('#deleteImgBtn').on('click', function() {
    $('#previewImg').attr('src', '');
    $('#fileName').text('');
    $('#previewArea').hide();
    $('#uploadZone').show();
    $('#analyzeBtn').prop('disabled', true);
    $('#imageInput').val('');
    imageUploaded = false;
    $('#uploadError').hide();
    $('#resultSection').hide();
    $('#videoResultSection').hide();
  });

  function handleFiles(files) {
    if (!files || files.length === 0) return;

    imageUploaded = true;
    $('#uploadZone').hide();
    $('#analyzeBtn').prop('disabled', false);

    if (files.length === 1) {
      const reader = new FileReader();
      reader.onload = function (e) {
        const isTif = files[0].name.toLowerCase().endsWith('.tif') || files[0].name.toLowerCase().endsWith('.tiff');
        if (!isTif) {
          $('#previewImg').attr('src', e.target.result).show();
        } else {
          $('#previewImg').hide();
        }
        $('#fileName').text(files[0].name);
        $('#previewArea').show();
      };
      reader.readAsDataURL(files[0]);
    } else {
      $('#previewImg').hide();
      $('#fileName').text(`${files.length}개 파일 선택됨`);
      $('#previewArea').show();
    }
  }

  $('#analyzeBtn').on('click', function () {
    const files = $('#imageInput')[0].files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    $('#resultSection').hide();
    $('#videoResultSection').hide();
    $('#loadingSection').show();
    $('#analyzeBtn').prop('disabled', true);

    if (files.length === 1) {
      formData.append('image', files[0]);
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
          $('#avgIoU').text(res.avg_iou);
          $('#resultSection').show();
          $('#analyzeBtn').prop('disabled', false);
        },
        error: function () {
          $('#loadingSection').hide();
          alert('분석 중 오류가 발생했습니다.');
          $('#analyzeBtn').prop('disabled', false);
        }
      });
    } else {
      for (let f of files) formData.append('images', f);
      $.ajax({
        url: '/cellsam/analyze_video',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        xhrFields: { responseType: 'blob' },
        success: function (blob) {
          $('#loadingSection').hide();
          const url = URL.createObjectURL(blob);
          $('#resultVideo').attr('src', url);
          $('#videoResultSection').show();
          $('#analyzeBtn').prop('disabled', false);
        },
        error: function () {
          $('#loadingSection').hide();
          alert('영상 분석 중 오류가 발생했습니다.');
          $('#analyzeBtn').prop('disabled', false);
        }
      });
    }
  });

});