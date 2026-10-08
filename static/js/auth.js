/* ==========================================================================
   登录 / 注册页
   ========================================================================== */
'use strict';

(function () {
  const form = document.getElementById('auth-form');
  if (!form) return;

  const msg = document.getElementById('auth-msg');
  const submit = document.getElementById('auth-submit');
  const modeInput = form.querySelector('input[name="mode"]');
  const registerFields = form.querySelectorAll('[data-only="register"]');
  const nextUrl = new URLSearchParams(window.location.search).get('next');

  function setMode(mode) {
    modeInput.value = mode;
    document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === mode));
    registerFields.forEach((field) => { field.hidden = mode !== 'register'; });
    submit.textContent = mode === 'register' ? '注册并登录' : '登录';
    const password = document.getElementById('a-password');
    password.setAttribute('autocomplete', mode === 'register' ? 'new-password' : 'current-password');
    msg.textContent = '';
    msg.className = 'auth-msg';
  }

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => setMode(tab.dataset.tab));
  });
  setMode('login');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const mode = modeInput.value;
    const body = {
      username: document.getElementById('a-username').value.trim(),
      password: document.getElementById('a-password').value,
    };
    if (mode === 'register') {
      body.display_name = document.getElementById('a-display').value.trim();
      body.contact = document.getElementById('a-contact').value.trim();
    }

    submit.disabled = true;
    submit.textContent = mode === 'register' ? '注册中…' : '登录中…';
    try {
      await LF.api(mode === 'register' ? '/api/auth/register' : '/api/auth/login', { method: 'POST', body });
      msg.textContent = '成功，正在跳转…';
      msg.className = 'auth-msg ok';
      LF.toast(mode === 'register' ? '注册成功' : '登录成功', 'ok');
      setTimeout(() => { window.location.href = nextUrl || '/'; }, 450);
    } catch (err) {
      msg.textContent = err.message;
      msg.className = 'auth-msg error';
      submit.disabled = false;
      submit.textContent = mode === 'register' ? '注册并登录' : '登录';
    }
  });
})();
