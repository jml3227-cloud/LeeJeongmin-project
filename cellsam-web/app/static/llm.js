$(document).ready(function () {

  // 전송 버튼 클릭
  $('#sendBtn').on('click', function () {
    sendMessage();
  });

  // 엔터키 전송 (shift+enter는 줄바꿈)
  $('#userInput').on('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function sendMessage() {
    const userText = $('#userInput').val().trim();
    if (!userText) return;

    appendMessage('user', userText);
    $('#userInput').val('');
    $('#loadingSection').show();
    $('#sendBtn').prop('disabled', true);

    $.ajax({
      url: '/llm/chat',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ message: userText }),
      success: function (res) {
        appendMessage('assistant', res.reply);
      },
      error: function () {
        appendMessage('assistant', '오류가 발생했습니다. 다시 시도해주세요.');
      },
      complete: function () {
        $('#loadingSection').hide();
        $('#sendBtn').prop('disabled', false);
      }
    });
  }

  function appendMessage(role, text) {
    const isUser = role === 'user';
    const align = isUser ? 'text-right' : 'text-left';
    const bgColor = isUser ? '#007bff' : '#f1f3f4';
    const textColor = isUser ? 'white' : '#333';
    const label = isUser ? '나' : 'AI';

    const html = `
      <div class="mb-3 ${align}">
        <p class="small text-muted mb-1">${label}</p>
        <div style="display:inline-block; max-width:75%; padding:10px 14px; border-radius:12px; background:${bgColor}; color:${textColor}; text-align:left; white-space:pre-wrap;">
          ${text}
        </div>
      </div>
    `;

    // 첫 메시지면 안내 문구 제거
    if ($('#chatHistory .text-center').length) {
      $('#chatHistory').empty();
    }

    $('#chatHistory').append(html);
    $('#chatHistory').scrollTop($('#chatHistory')[0].scrollHeight);
  }

});