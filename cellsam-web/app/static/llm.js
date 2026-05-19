document.addEventListener('DOMContentLoaded', function () {

  document.getElementById('sendBtn').addEventListener('click', function () {
    sendMessage();
  });

  document.getElementById('userInput').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function sendMessage() {
    const userText = document.getElementById('userInput').value.trim();
    if (!userText) return;

    appendMessage('user', userText);
    document.getElementById('userInput').value = '';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('sendBtn').disabled = true;

    fetch('/llm/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userText })
    })
    .then(res => res.json())
    .then(res => {
      appendMessage('assistant', res.reply);
    })
    .catch(() => {
      appendMessage('assistant', '오류가 발생했습니다. 다시 시도해주세요.');
    })
    .finally(() => {
      document.getElementById('loadingSection').style.display = 'none';
      document.getElementById('sendBtn').disabled = false;
    });
  }

  function appendMessage(role, text) {
    const isUser = role === 'user';
    const align = isUser ? 'text-right' : 'text-left';
    const bgColor = isUser ? '#007bff' : '#f1f3f4';
    const textColor = isUser ? 'white' : '#333';
    const label = isUser ? '나' : 'AI';

    const div = document.createElement('div');
    div.className = `mb-3 ${align}`;
    div.innerHTML = `
      <p class="small text-muted mb-1">${label}</p>
      <div style="display:inline-block; max-width:75%; padding:10px 14px; border-radius:12px; background:${bgColor}; color:${textColor}; text-align:left; white-space:pre-wrap; text-indent:0;">${text}</div>
    `;

    const chatHistory = document.getElementById('chatHistory');
    const placeholder = chatHistory.querySelector('.text-center');
    if (placeholder) chatHistory.innerHTML = '';

    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
});